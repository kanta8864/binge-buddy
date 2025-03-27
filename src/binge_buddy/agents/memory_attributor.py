import ast
import json
import logging
import re
from typing import List

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.agent_state.states import AgentState
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.memory import Memory
from binge_buddy.ollama import OllamaLLM

# Configure logging (if not already configured elsewhere)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MemoryAttributor(BaseAgent):
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
                You are a highly accurate memory attributor.

                Your job is to **assign an appropriate attribute** to each piece of extracted memory.

                ### **Task Overview**  
                You will receive a list of memory strings. Each memory contains a fact about a user's movie preferences. Your task is to categorize each memory under the most appropriate **attribute** from the predefined list.

                ### **Output Requirements (STRICT)**  
                - The output **must** be a valid Python list of dictionaries.  
                - Each dictionary **must** have exactly two keys:  
                  - `"information"` → The original memory string.  
                  - `"attribute"` → The corresponding attribute from the predefined list.  
                - No extra text, explanations, or formatting—only the structured Python list.

                ### **Attributes to Choose From:**  
                Every memory **must** be categorized into one of the following attributes:

                - **LIKES:** Movies and genres the user likes.  
                - **DISLIKES:** Movies and genres the user dislikes.  
                - **FAVORITE:** Favorite movies.  
                - **WANTS_TO_WATCH:** Movies the user wants to watch.  
                - **PLATFORM:** Preferred streaming platforms.  
                - **GENRE:** Preferred genres.  
                - **PERSONALITY:** The user’s personality regarding movies.  
                - **WATCHING_HABIT:** Watching habits.  
                - **FREQUENCY:** How often the user watches movies/shows.  
                - **AVOID:** Categories the user avoids.  
                - **CHARACTER_PREFERENCES:** Preferred character types.  
                - **SHOW_LENGTH:** Preferred show length.  
                - **REWATCHER:** Whether the user enjoys rewatching content.  
                - **POPULARITY:** Preference for mainstream vs. niche content.  

                ### **Input Example**  
                ```python
                [
                    "Likes psychological thrillers",
                    "Watches mostly on Netflix",
                    "Prefers witty and sarcastic characters"
                ]
                Correct Output Format
                ```python
                [
                    {{"information": "Likes psychological thrillers", "attribute": "GENRE"}},
                    {{"information": "Watches mostly on Netflix", "attribute": "PLATFORM"}},
                    {{"information": "Prefers witty and sarcastic characters", "attribute": "CHARACTER_PREFERENCES"}}
                ]
                """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(
                    variable_name="extracted_memories"
                ),  # Holds the list of extracted memories
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_aggregator_runnable = self.prompt | self.llm_runnable

    def format_memories(self, response: str, state: AgentState) -> List[Memory]:
        # Extract the part inside brackets using regex
        match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)

        if not match:
            return []  # Return empty list if no valid list is found

        memories_string = match.group(0).strip()  # Extract matched text

        try:
            # Use ast.literal_eval to safely parse the list of dictionaries
            memories_list = ast.literal_eval(memories_string)

            # Ensure it's a valid list of dictionaries with correct structure
            if not isinstance(memories_list, list) or not all(
                isinstance(mem, dict) and "information" in mem and "attribute" in mem
                for mem in memories_list
            ):
                return []

            # Convert extracted memories into Memory objects
            return [
                Memory.create(
                    information=mem["information"].strip(),
                    attribute=mem["attribute"].strip(),
                    state=state,
                )
                for mem in memories_list
            ]

        except (SyntaxError, ValueError):
            return []  # Return empty list if parsing fails

    def process(self, state: AgentState) -> AgentState:

        messages = {}

        assert (
            state.extracted_memories
        ), "No extracted memories found by memory attributer"
        messages["extracted_memories"] = [
            AIMessage(
                content=json.dumps(
                    [memory.information for memory in state.extracted_memories]
                )
            )
        ]

        response = self.memory_aggregator_runnable.invoke(messages)

        response = utils.remove_think_tags(response)

        memories_with_attributes = self.format_memories(response, state)

        # Attempt to run the pipeline once again
        if not memories_with_attributes:
            response = self.memory_aggregator_runnable.invoke(messages)

            response = utils.remove_think_tags(response)

            memories_with_attributes = self.format_memories(response, state)

        logging.info(f"Memory Attributor response: {response}")

        logging.info(f"Attributed Memories: {memories_with_attributes}")

        state.extracted_memories = memories_with_attributes

        return state
