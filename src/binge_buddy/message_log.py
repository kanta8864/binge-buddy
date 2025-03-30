import logging
import threading
from typing import Callable, Iterator, List, Optional

from langchain.schema import HumanMessage

from binge_buddy.agent_state.states import (
    AgentState,
    EpisodicAgentState,
    SemanticAgentState,
)
from binge_buddy.memory_workflow.episodic_workflow import EpisodicWorkflow
from binge_buddy.memory_workflow.semantic_workflow import SemanticWorkflow
from binge_buddy.message import Message, UserMessage


class MessageLog:
    def __init__(self, user_id, session_id, memory_handler, mode):
        self.user_id = user_id
        self.session_id = session_id
        self.messages: List[Message] = []
        self.subscribers: List[Callable[[AgentState], None]] = []
        self.memory_handler = memory_handler
        self.mode = mode

        if self.mode == "semantic":
            workflow = SemanticWorkflow(memory_handler)
        else:
            workflow = EpisodicWorkflow(memory_handler)

        self.subscribe(workflow.run)

    def add_message(self, message: Message):
        self.messages.append(message)
        if isinstance(message, UserMessage):
            self.notify_subscribers(message)

    def subscribe(self, callback: Callable[[Message], None]):
        self.subscribers.append(callback)

    def notify_subscribers(self, message: UserMessage):
        logging.info("notifying subscribers")
        for subscriber in self.subscribers:
            memories = self.memory_handler.get_existing_memories(message.user_id)
            if self.mode == "semantic":
                state = SemanticAgentState(
                    user_id=message.user_id,
                    existing_memories=memories,
                    current_user_message=message,
                )
            else:
                state = EpisodicAgentState(
                    user_id=message.user_id,
                    existing_memories=memories,
                    current_user_message=message,
                )

            thread = threading.Thread(target=subscriber, args=(state,))
            thread.start()

    def get_last_message(self) -> Optional[Message]:
        """
        Returns the last message content in the session log, including its role.

        :return: The last message content along with its role or None if no messages exist.
        """
        if self.messages:
            return self.messages[-1]
        return None

    def clear_log(self) -> None:
        """
        Clears the message log (resets the log for a new session or scenario).
        """
        self.messages.clear()

    def get_message_count(self) -> int:
        """
        Get the number of messages in the current session.

        :return: The number of messages in the log.
        """
        return len(self.messages)

    def __len__(self) -> int:
        """
        Allows using len() on the MessageLog to get the number of messages.

        :return: The number of messages in the log.
        """
        return len(self.messages)

    def __str__(self) -> str:
        """
        Return a string representation of the message log, useful for debugging.

        :return: A string representation of the log with roles and contents.
        """
        return "\n".join(
            f"{message.role}: {message.content}" for message in self.messages
        )

    def __iter__(self) -> Iterator[Message]:
        """
        Returns a new iterator for the messages each time __iter__ is called.
        """
        return MessageLogIterator(self.messages)


class MessageLogIterator:
    def __init__(self, messages: List[Message]):
        """
        Initialize the iterator with a list of messages.
        """
        self.messages = messages
        self._index = 0  # Initialize index to start from the first message

    def __iter__(self):
        """
        Return the iterator itself.
        """
        return self

    def __next__(self) -> Message:
        """
        Return the next message in the list, or raise StopIteration if done.
        """
        if self._index < len(self.messages):
            message = self.messages[self._index]
            self._index += 1
            return message
        else:
            raise StopIteration
