import json
import logging
import re

from langchain.llms.base import LLM
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.agent_state.states import AgentState, SemanticAgentState
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.message import AgentMessage
from binge_buddy.ollama import OllamaLLM


class AggregatorReviewer(BaseAgent):
    def __init__(self, llm: LLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
                You are an expert memory reviewer tasked with ensuring the integrity of a user's long-term memory.  
                Your role is to **critically evaluate** the **aggregated memories** produced by the system and verify that they are:  

                1. **Accurate** - They correctly reflect both the existing and newly extracted memories.  
                2. **Complete** - No key details are missing.  
                3. **Consistent** - They do not contradict the user's prior knowledge.  
                4. **Non-hallucinatory** - No fabricated or assumed details have been introduced.  

                ---

                ### **Review Process**
                Carefully analyze the **aggregated memories** against the **existing** and **newly extracted** memories using the following checks:

                #### **1. Check for Hallucinations**  
                - Ensure all details in **aggregated memories** are directly **traceable** to either **existing memories** or **newly extracted memories**.  
                - If any information was added that **does not exist in either source**, flag it.  

                #### **2. Check for Completeness**  
                - Verify that **all key details** from **existing** and **new memories** are present.  
                - If any meaningful information has been **omitted**, flag it.  

                #### **3. Check for Incorrect Aggregation**  
                - Ensure no information has been **distorted, misrepresented, or incorrectly merged**.  
                - If an **existing memory was wrongly altered** or **lost important context**, flag it.  

                ---

                ### **Memory Data Provided**
                - **Existing Memories:** `{existing_memories}`
                - **Newly Extracted Memories:** `{extracted_memories}`
                - **Aggregated Memories:** `{aggregated_memories}`

                ---

                ### **Response Format**
                You must **strictly** use one of the following responses:

                If the aggregated memories are **fully correct**, respond with:  
                APPROVED

                If the aggregated memories contain **errors**, respond with:  

                REJECTED 
                REPAIR MESSAGE: 
                [Clearly explain why the aggregated memories need correction.]

                ---

                ### **Examples**
                #### **Example 1 (Correct Aggregation)**
                **Input Aggregated Memories:**
                [ 
                 {{
                     "information": "Likes psychological thrillers", "attribute": "GENRE"
                 }}, 
                 {{
                     "information": "Watches mostly on Netflix", "attribute": "PLATFORM"
                 }} 
                ]
                **Response:**  
                APPROVED

                #### **Example 2 (Incorrect Aggregation - Missing Data)**
                **Input Aggregated Memories:**
                [
                    {{
                        "information": "Likes psychological thrillers", "attribute": "GENRE"
                    }} 
                ]
                **Response:**  

                REJECTED 
                REPAIR MESSAGE: 
                "The 'PLATFORM' memory is missing from the aggregation. Ensure all relevant information is included."

                ---

                ### **Important Notes**
                 **You MUST follow the exact response format.**  
                 **No extra explanations or formatting—only "APPROVED" or "REJECTED" with a clear repair message if needed.**  
                 **Your decision must be precise, ensuring the final memory is both accurate and complete. You don't have to be very strict and can be a little lenient.**  

                ---
            """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="existing_memories", optional=True),
                MessagesPlaceholder(variable_name="extracted_memories"),
                MessagesPlaceholder(variable_name="aggregated_memories"),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.aggregator_reviewer_runnable = self.prompt | self.llm_runnable

    def parse_model_output(self, response):
        response = response.strip()

        # Normalize response case
        normalized_response = response.upper()

        # Check if the response is APPROVED
        if normalized_response == "APPROVED":
            return {"status": "APPROVED", "message": None}

        # Check if the response is REJECTED with a repair message
        match = re.search(
            r"REPAIR MESSAGE:\s*(.*)", response, re.DOTALL | re.IGNORECASE
        )
        if normalized_response.startswith("REJECTED") and match:
            repair_message = match.group(1).strip()
            return {"status": "REJECTED", "message": repair_message}

        # If the format is unrecognized, return as UNKNOWN
        return {"status": "UNKNOWN", "message": response}

    def process(self, state: AgentState) -> AgentState:
        if not isinstance(state, SemanticAgentState):
            raise TypeError(f"Expected SemanticAgentState, got {type(state).__name__}")

        messages = {}

        messages["extracted_memories"] = [
            AIMessage(
                content=json.dumps(
                    [memory.as_dict() for memory in state.extracted_memories]
                )
            )
        ]

        messages["aggregated_memories"] = [
            AIMessage(
                content=json.dumps(
                    [memory.as_dict() for memory in state.aggregated_memories]
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

        response = self.aggregator_reviewer_runnable.invoke(messages)
        response = utils.remove_think_tags(response)

        parsed_output = self.parse_model_output(response)

        if parsed_output["status"] == "APPROVED":
            state.needs_repair = False
            state.retry_count = 0
        else:
            state.needs_repair = True
            state.repair_message = AgentMessage(
                content=parsed_output["message"],
                user_id=state.user_id,
                session_id=state.current_user_message.session_id,
            )

        logging.info(f"Aggregator Reviewer Response: {parsed_output['status']}")
        logging.info(f"Aggregator Reviewer Reasoning: {parsed_output['message']}")

        return state
