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
from binge_buddy.agent_state.states import (
    AgentState,
    AgentStateDict,
    SemanticAgentState,
)
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.memory import Memory, SemanticMemory
from binge_buddy.ollama import OllamaLLM

# Configure logging (if not already configured elsewhere)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MemoryAggregator(BaseAgent):
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
                You are a supervisor responsible for aggregating user movie preference memories into a structured format.

                ## **Given Information:**
                - **Extracted Memories**: `{extracted_memories}` (new memories, each with an attribute).
                - **Existing Memories**: `{existing_memories}` (previously stored memories, also categorized).

                ### **Your Task:**
                1. **Group memories by their attributes** (e.g., all "GENRE" memories should be combined).  
                2. **Aggregate new and existing memories without losing information**:  
                   - If **existing memories are present**, **merge** them with the extracted memories, ensuring each attribute is represented **only once**.  
                   - If **no existing memories are present**, return the extracted memories as is.  
                   - Ensure all relevant details are preserved and combined logically.  
                   - Do **not** duplicate identical pieces of information.  
                3. **Output the final aggregated memories in the required structured format.**  

                ### **Correct Output Format**
                Your response must strictly follow this format:
                ```python
                [
                    {{
                        "information": "Likes psychological thrillers and mind-bending plots", "attribute": "GENRE"
                    }},
                    {{
                        "information": "Watches mostly on Netflix and occasionally on Hulu", "attribute": "PLATFORM"
                    }},
                    {{
                        "information": "Prefers witty and sarcastic characters but dislikes overly serious protagonists", "attribute": "CHARACTER_PREFERENCES"
                    }}
                ]
                Important Rules:

                - If existing memories are empty, return extracted memories as is.
                - Each attribute appears only once in the final list.
                - Information is merged meaningfully into a single "information" field per attribute.
                - No extra formatting, explanations, or bullet points—just a valid Python list of dictionaries.
                - Your goal is to return only the correctly formatted aggregated list. Any deviation from this will be considered incorrect.
                - Even if you are repairing a message, the response should only be in the described format. No extra test or helpful message. Remember this.
                - Do not respond with any solution code, I only want the response in the output format described. Only in that format.

                ### **Correct Output Format**
                Your response must strictly follow this format:
                ```python
                [
                    {{
                        "information": "Likes psychological thrillers and mind-bending plots", "attribute": "GENRE"
                    }},
                    {{
                        "information": "Watches mostly on Netflix and occasionally on Hulu", "attribute": "PLATFORM"
                    }},
                    {{
                        "information": "Prefers witty and sarcastic characters but dislikes overly serious protagonists", "attribute": "CHARACTER_PREFERENCES"
                    }}
                ]
            """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="existing_memories", optional=True),
                MessagesPlaceholder(variable_name="extracted_memories"),
                MessagesPlaceholder(
                    variable_name="repair_message", optional=True
                ),  # Holds the repair message if present
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

        if state.state_type != "semantic":
            return state

        messages = {}

        if state.extracted_memories is None:
            return state

        messages["extracted_memories"] = [
            AIMessage(
                content=json.dumps(
                    [memory.as_dict() for memory in state.extracted_memories]
                )
            )
        ]

        if state.existing_memories:
            messages["existing_memories"] = [
                AIMessage(
                    content=json.dumps(
                        [memory.as_dict() for memory in state.existing_memories]
                    )
                )
            ]

        else:
            messages["existing_memories"] = []

        if state.needs_repair and state.repair_message is not None:
            if state.retry_count > 20:
                state.aggregated_memories = []
                return state

            messages["repair_message"] = [state.repair_message.to_langchain_message()]

            state.retry_count += 1
            state.needs_repair = False
            state.repair_message = None

        response = self.memory_aggregator_runnable.invoke(messages)

        response = utils.remove_think_tags(response)

        aggregated_memories_with_attributes = self.format_memories(response, state)

        logging.info(f"Memory Aggregator response: {response}")

        logging.info(f"Aggregated Memories: {aggregated_memories_with_attributes}")

        assert isinstance(
            state, SemanticAgentState
        ), f"Expected SemanticAgentState, found {type(state)}"

        # assert all(
        #     isinstance(mem, SemanticMemory)
        #     for mem in aggregated_memories_with_attributes
        # ), "Expected all aggregated memories to be SemanticMemories, found some that aren't"

        state.aggregated_memories = aggregated_memories_with_attributes

        return state
