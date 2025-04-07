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
You are an expert memory reviewer ensuring that aggregated user memories are reasonably accurate, broadly complete, and free from major issues.

---

### ✅ APPROVED if:
- The information roughly matches what's in the existing or new memories.
- No **major** details are missing.
- No obvious contradictions or hallucinations.
- Each `attribute` appears only once across aggregated memories.

⏳ **Be generous** with approval. Slight paraphrasing, soft merges, or vague language are fine as long as the memory makes sense and nothing important is clearly wrong.

---

### ❌ Only REJECT if:
- There are **multiple entries with the same attribute** that were not merged and clearly should have been.
- Important memories are totally missing or obviously incorrect.
- There are clear fabrications (information that appears in neither existing nor new memories).

---

### Attribute Merge Clarification:

- ❌ Incorrect (do not allow this):
  [
    SemanticMemory(information="Likes action movies", attribute="LIKES"),
    SemanticMemory(information="Likes scifi movies", attribute="LIKES")
  ]

- ✅ Correct (allowed, even if phrased differently):
  [
    SemanticMemory(information="Likes action movies and scifi movies", attribute="LIKES")
  ]

---

### Memory Data
- Existing Memories: `{existing_memories}`
- Newly Extracted Memories: `{extracted_memories}`
- Aggregated Memories: `{aggregated_memories}`

---

### Response Format
If it's mostly fine:  
APPROVED

If there's a clear issue:  
REJECTED  
REPAIR MESSAGE:  
[Briefly explain the issue — only if it's serious.]

Be relaxed and trust small imperfections. Only flag things if they truly break the rules.
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
