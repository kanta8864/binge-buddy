from abc import ABC
from typing import List, Optional, TypedDict

from binge_buddy.memory import Memory, MemoryDict, SemanticMemory
from binge_buddy.message import Message, MessageDict, UserMessage


class AgentStateDict(TypedDict, total=False):  # total=False makes all fields optional
    user_id: str
    existing_memories: List[MemoryDict]
    current_user_message: MessageDict
    contains_information: Optional[bool]
    extracted_memories: Optional[List[MemoryDict]]
    needs_repair: Optional[bool]
    repair_message: Optional[Message]
    retry_count: int


class AgentState(ABC):
    def __init__(
        self,
        state_type: str,
        user_id: str,
        existing_memories: List[Memory],
        current_user_message: Message,
        contains_information: Optional[bool] = None,
        extracted_memories: Optional[List[Memory]] = None,
        needs_repair: Optional[bool] = None,
        repair_message: Optional[Message] = None,
        retry_count: int = 0,
    ):
        self.user_id = user_id
        self.existing_memories = existing_memories
        self.current_user_message = current_user_message
        self.contains_information = contains_information
        self.extracted_memories = extracted_memories
        self.needs_repair = needs_repair
        self.repair_message = repair_message
        self.state_type = state_type
        self.retry_count = retry_count

    def as_dict(self) -> AgentStateDict:
        """Convert state object to dictionary for LangGraph compatibility."""
        return {
            "user_id": self.user_id,
            "existing_memories": [mem.as_dict() for mem in self.existing_memories],
            "current_user_message": self.current_user_message.as_dict(),
            "contains_information": self.contains_information,
            "extracted_memories": (
                [mem.as_dict() for mem in self.extracted_memories]
                if self.extracted_memories
                else None
            ),
            "needs_repair": self.needs_repair,
            "repair_message": self.repair_message.as_dict(),
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: AgentStateDict) -> "AgentState":
        """Reconstructs an AgentState object from a dictionary."""
        return cls(
            user_id=data["user_id"],
            existing_memories=[Memory.from_dict(m) for m in data["existing_memories"]],
            current_user_message=UserMessage.from_dict(data["current_user_message"]),
            contains_information=data.get("contains_information"),
            extracted_memories=(
                [Memory.from_dict(m) for m in data["extracted_memories"]]
                if data.get("extracted_memories")
                else None
            ),
            needs_repair=data.get("needs_repair"),
            repair_message=data.get("repair_message"),
            retry_count=data.get("retry_count"),
        )


class SemanticAgentStateDict(AgentStateDict):
    aggregated_memories: Optional[List[MemoryDict]]


class SemanticAgentState(AgentState):
    aggregated_memories: Optional[List[Memory]] = None

    def __init__(
        self,
        user_id: str,
        existing_memories: List[Memory],
        current_user_message: UserMessage,
    ):
        super().__init__(
            user_id=user_id,
            existing_memories=existing_memories,
            current_user_message=current_user_message,
            state_type="semantic",
        )
        self.aggregated_memories = None

    def as_dict(self) -> SemanticAgentStateDict:
        """Extend `as_dict()` to include `aggregated_memories`."""
        base_dict = super().as_dict()
        base_dict["aggregated_memories"] = (
            [mem.as_dict() for mem in self.aggregated_memories]
            if self.aggregated_memories
            else None
        )
        return base_dict


class EpisodicAgentState(AgentState):
    def __init__(
        self,
        user_id: str,
        existing_memories: List[Memory],
        current_user_message: UserMessage,
    ):
        super().__init__(
            user_id=user_id,
            existing_memories=existing_memories,
            current_user_message=current_user_message,
            state_type="episodic",
        )

    def as_dict(self) -> AgentStateDict:
        base_dict = super().as_dict()
        return base_dict
