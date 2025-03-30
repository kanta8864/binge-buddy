import ast
import json
import logging
import re
from typing import Dict, List, Optional

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.agent_state.states import AgentState, AgentStateDict
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.memory import EpisodicMemory, Memory, SemanticMemory
from binge_buddy.message import AgentMessage
from binge_buddy.ollama import OllamaLLM


class MemoryExtractor(BaseAgent):
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
                You are a supervisor managing a team of movie recommendation experts.

                Your team's job is to build the perfect knowledge base about a user's movie preferences in order to provide highly personalized recommendations.

                The knowledge base should ultimately consist of many discrete pieces of information that add up to a rich persona (e.g., Likes action movies; Dislikes horror; Favorite movie is Inception; Watches mostly on Netflix; Enjoys long TV series; Prefers witty and sarcastic characters, etc.).

                Every time you receive a message, you will evaluate whether it contains any information worth recording in the knowledge base.

                A message may contain multiple pieces of information that should be saved separately.

                ---

                ### **Types of Information to Track**  
                You are only interested in the following categories of information:

                - **LIKES:** Movies and Genres the user likes  
                - **DISLIKES:** Movies and Genres the user dislikes  
                - **FAVORITE:** Favorite movies  
                - **WANTS_TO_WATCH:** Movies the user wants to watch  
                - **PLATFORM:** Preferred streaming platforms  
                - **GENRE:** Preferred genres  
                - **PERSONALITY:** User's personality in relation to movies  
                - **WATCHING_HABIT:** Watching habits  
                - **FREQUENCY:** Frequency of watching  
                - **AVOID:** Categories the user wants to avoid  
                - **CHARACTER_PREFERENCES:** Character archetypes the user enjoys  
                - **SHOW_LENGTH:** Preferences for show length  
                - **REWATCHER:** Whether the user enjoys rewatching movies/shows  
                - **POPULARITY:** Preference for mainstream vs. niche content  

                ---

                ### **How You Should Process Messages**

                Each time you receive a message, follow these steps:

                #### **1. Extract New Memories**
                - Analyze the latest **user message** for new information.  
                - Identify **all relevant details** that belong to one of the above categories.  
                - If multiple pieces of information are found, list them separately.  

                ---

                ### **2. Repair Memories (If a Repair Request is Present)**  
                Sometimes, you will receive a **repair request**. This means that some of the previously extracted memories from the same user message were incorrect.  

                In this case, you will receive:  
                1. **The original extracted memories** (as a list of strings).  
                2. **A repair message** explaining what needs to be corrected.  

                #### **Steps for Repairing Memories:**  
                - Carefully analyze the repair message.  
                - Identify errors or inaccuracies in the provided memories.  
                - Correct the memories based on the repair instructions.  
                - Ensure the final memories are **fully accurate and aligned** with the user’s actual preferences.  

                ---

                ### **Final Output Format (STRICT REQUIREMENT)**  

                **YOU MUST STRICTLY FOLLOW THIS OUTPUT FORMAT. FAILURE TO DO SO WILL RESULT IN REJECTION.** 🚨

                #### **Correct Format:**
                A valid Python list of strings, like this:

                ```python
                [
                    "...",
                    "...",
                    "..."
                ]
                ```

                Incorrect Formats (DO NOT USE THESE):
                Do not use dashes, bullets, or extra characters:
                - "Likes action movies"
                - "Dislikes horror"
                - FINAL WARNING: ONLY return a valid Python list of strings and nothing else. Any deviation from this format is an error. Even if you are repairing a message,
                the response should only return a valid Python list of strings without any extra labels, dashes, attributes, or helpful message. Remember this.
         """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(
                    variable_name="current_user_message"
                ),  # Holds the latest user message
                MessagesPlaceholder(
                    variable_name="repair_message", optional=True
                ),  # Holds the repair message if present
                MessagesPlaceholder(
                    variable_name="memories_to_repair", optional=True
                ),  # Holds extracted memories if repair mode is active
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_extractor_runnable = self.prompt | self.llm_runnable

    def format_memories(self, response: str, state: AgentState) -> List[Memory]:
        # Extract the part inside brackets using regex (handling newlines properly)
        match = re.search(r"\[\s*([\s\S]*?)\s*\]", response)

        if not match:
            return []  # Return empty list if no valid list is found

        memories_string = match.group(0).strip()  # Extract matched text

        try:
            # Clean the string to ensure proper parsing
            cleaned_memories_string = memories_string.replace("\n", "").strip()

            # Use ast.literal_eval to safely parse the list
            memories_list = ast.literal_eval(cleaned_memories_string)

            # Ensure it's a valid list of strings, stripping extra spaces
            if not isinstance(memories_list, list) or not all(
                isinstance(mem, str) for mem in memories_list
            ):
                return []

            # Convert extracted memories into Memory objects, trimming whitespace
            return [
                Memory.create(information=mem.strip(), state=state)
                for mem in memories_list
            ]

        except (SyntaxError, ValueError) as e:
            logging.error(
                f"Memory parsing error: {str(e)}"
            )  # Optional logging for debugging
            return []  # Return empty list if parsing fails

    def process(self, state: AgentState) -> AgentState:

        messages = {
            "current_user_message": [state.current_user_message.to_langchain_message()]
        }

        if (
            state.needs_repair
            and state.extracted_memories
            and state.repair_message is not None
        ):
            if state.retry_count > 20:
                state.extracted_memories = []
                return state

            messages["repair_message"] = [state.repair_message.to_langchain_message()]
            messages["memories_to_repair"] = [
                AIMessage(
                    content=json.dumps(
                        [memory.information for memory in state.extracted_memories]
                    )
                )
            ]

            state.retry_count += 1
            state.needs_repair = False
            state.repair_message = None

        # Run the pipeline and get the response
        response = utils.remove_think_tags(
            self.memory_extractor_runnable.invoke(messages)
        )
        response = response.split("Memory Extractor Result:", 1)[-1].strip()

        memories = self.format_memories(response, state)

        if not memories:
            # Attempt to run the pipeline again just to make sure it's not a parsing error
            response = utils.remove_think_tags(
                self.memory_extractor_runnable.invoke(messages)
            )
            response = response.split("Memory Extractor Result:", 1)[-1].strip()

            memories = self.format_memories(response, state)

        # Log the extracted response
        logging.info(f"Memory Extractor response: {response}")
        logging.info(f"Extracted Memories: {memories}")

        state.extracted_memories = memories

        return state
