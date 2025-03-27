import json
import logging
import re

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
from binge_buddy.memory import SemanticMemory
from binge_buddy.message import AgentMessage
from binge_buddy.ollama import OllamaLLM


class ExtractorReviewer(BaseAgent):
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
                You are an expert memory reviewer.

                Your job is to **verify** the newly extracted memories to ensure they are:
                1. **Accurate** – They must correctly reflect the user's message.
                2. **Complete** – They must include all relevant information from the user's message.
                3. **Non-hallucinatory** – They must not introduce information that was never mentioned.

                ---
                ### **Review Process**
                1. **Check for Hallucinations**  
                   - If any extracted memory contains information **not explicitly stated** in the user's message, a repair is needed.  
                   
                2. **Check for Completeness**  
                   - If the extracted memories **fail to capture important details** from the user's message, a repair is needed.  

                ---
                ### **Inputs**
                #### **User's Message**
                {current_user_message}

                #### **Extracted Memories**
                {extracted_memories}

                ---
                ### **Output Format**
                If all extracted memories are **correct and complete**, respond with:
                APPROVED

                If any extracted memory **needs fixing**, respond with:
                REJECTED
                REPAIR MESSAGE: [Provide a clear and helpful message explaining why the memories need to be regenerated.]
                ---

                ### **Examples**
                current_user_message:
                "I love watching psychological thrillers like Inception and Shutter Island, but I can't stand horror movies. I usually watch on Netflix."

                extracted_memories:
                [
                    "Likes psychological thrillers.",
                    "Favorite movie is Inception.",
                    "Watches on Netflix."
                ]

                REJECTED
                REPAIR MESSAGE:
                The extracted memories are incomplete. While they correctly capture the user's preference for psychological thrillers and Netflix, they **fail to include the user's dislike for horror movies** and their mention of Shutter Island. Please regenerate the memories with full accuracy.
                ---

                Remember, only respond using the output format specified and nothing else. Do not include any additional explanations, reasoning, or commentary outside of the specified format.
                """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="current_user_message"),
                MessagesPlaceholder(variable_name="extracted_memories"),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_reviewer_runnable = self.prompt | self.llm_runnable

    def parse_model_output(self, response):
        response = response.strip()

        # Check if the response is APPROVED
        if response == "APPROVED":
            return {"status": "APPROVED", "message": None}

        # Check if the response is REJECTED with a repair message
        match = re.search(r"REPAIR MESSAGE:\s*(.*)", response, re.DOTALL)
        if response.startswith("REJECTED") and match:
            repair_message = match.group(1).strip()
            return {"status": "REJECTED", "message": repair_message}

        # If the format is unrecognized, return as UNKNOWN
        return {"status": "UNKNOWN", "message": response}

    def process(self, state: AgentState) -> AgentState:

        messages = {}

        messages["current_user_message"] = [
            state.current_user_message.to_langchain_message()
        ]

        messages["extracted_memories"] = [
            AIMessage(
                content=json.dumps(
                    [memory.information for memory in state.extracted_memories]
                )
            )
        ]
        response = utils.remove_think_tags(
            self.memory_reviewer_runnable.invoke(messages)
        )

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

        logging.info(f"Extractor Reviewer Response: {parsed_output['status']}")
        logging.info(f"Extractor Reviewer Reasoning: {parsed_output['message']}")

        return state
