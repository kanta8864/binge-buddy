from datetime import datetime
from typing import TypedDict, Union

from langchain.schema import AIMessage, BaseMessage, HumanMessage
from pydantic import Field


class MessageDict(TypedDict):
    content: str
    role: str
    user_id: str
    session_id: str
    timestamp: str
    type: str


class Message(BaseMessage):
    role: str = Field(..., description="The role of the sender (user/agent).")
    user_id: str = Field(..., description="User ID associated with the message.")
    session_id: str = Field(..., description="Session ID for the conversation.")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Message timestamp."
    )
    type: str

    def to_langchain_message(self) -> Union[HumanMessage, AIMessage]:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def from_langchain_message(
        lc_message: Union[HumanMessage, AIMessage],
        user_id: str,
        session_id: str,
    ) -> "Message":
        role = "user" if isinstance(lc_message, HumanMessage) else "system"
        return Message(
            content=lc_message.content,
            role=role,
            user_id=user_id,
            session_id=session_id,
            type=role,
        )

    def as_dict(self) -> MessageDict:
        """Custom method to return a dictionary representation of the object."""
        return {
            "content": self.content,
            "role": self.role,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: MessageDict) -> "Message":
        """Dynamically create a UserMessage or Message based on role."""
        if data["role"] == "user":
            return UserMessage(
                content=data["content"],
                user_id=data["user_id"],
                session_id=data["session_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )
        return cls(
            content=data["content"],
            role=data["role"],
            user_id=data["user_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            type=data["type"],
        )

    def __repr__(self):
        return f"Message(user_id={self.user_id}, session_id={self.session_id}, role={self.role}, timestamp={self.timestamp})"

    def __str__(self):
        return f"Message(role:{self.role}, content:{self.content})"


class UserMessage(Message):
    role: str = "user"
    type: str = "user"

    def to_langchain_message(self) -> Union[HumanMessage, AIMessage]:
        return HumanMessage(content=self.content)

    def __repr__(self):
        return f"UserMessage(user_id={self.user_id}, session_id={self.session_id}, content={self.content})"

    def __str__(self):
        return f"UserMessage(role:{self.role}, content:{self.content})"


class AgentMessage(Message):
    role: str = "system"
    type: str = "system"

    def to_langchain_message(self) -> Union[HumanMessage, AIMessage]:
        return AIMessage(content=self.content)

    def __repr__(self):
        return f"AgentMessage(user_id={self.user_id}, session_id={self.session_id}, content={self.content})"

    def __str__(self):
        return f"AgentMessage(role:{self.role}, content:{self.content})"
