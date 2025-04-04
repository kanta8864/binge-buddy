import logging
from abc import ABC, abstractmethod

from binge_buddy.agent_state.states import (
    AgentState,
    EpisodicAgentState,
    SemanticAgentState,
)
from binge_buddy.memory import EpisodicMemory, SemanticMemory
from binge_buddy.memory_db import MemoryDB


class MemoryHandler(ABC):
    """Base class for handling memory operations."""

    def __init__(self, memory_db: MemoryDB):
        self.memory_db = memory_db

    @abstractmethod
    def process(self, state: AgentState):
        """Process the given agent state."""
        ...

    @abstractmethod
    def get_existing_memories(self, user_id: str):
        """Process the given agent state."""
        ...


class SemanticMemoryHandler(MemoryHandler):
    """Handles processing of semantic memories."""

    collection_name = "semantic_memory"

    def process(self, state: SemanticAgentState):
        if not isinstance(state, SemanticAgentState):
            raise TypeError("Expected SemanticAgentState")

        collection = self.memory_db.get_collection(self.collection_name)
        user_id = state.user_id

        logging.info(f"Semantic Memory Handler: Adding memories...")
        for memory in state.aggregated_memories:
            if not memory.has_attribute():
                continue

            db_entry = memory.as_db_entry()  # {attribute: memory_str}
            attribute, memory_str = next(iter(db_entry.items()))

            # Update or insert attribute in user document
            collection.update_one(
                {"user_id": user_id},
                {"$set": {f"memory.{attribute}": memory_str}},
                upsert=True,
            )

            logging.info(f"memory.{attribute}: {memory_str}")

    def get_existing_memories(self, user_id):
        query = {"user_id": user_id}  # Query to find the user
        result = self.memory_db.find_one(
            self.collection_name, query
        )  # Fetch the document
        existing_memories = []
        if result:
            for attr, information in result["memory"].items():
                existing_memories.append(
                    SemanticMemory(information=information, attribute=attr)
                )

        return existing_memories


class EpisodicMemoryHandler(MemoryHandler):
    """Handles processing of episodic memories."""

    collection_name = "episodic_memory"

    def process(self, state: EpisodicAgentState):
        if not isinstance(state, EpisodicAgentState):
            raise TypeError("Expected EpisodicAgentState")

        collection = self.memory_db.get_collection(self.collection_name)
        user_id = state.user_id

        logging.info(f"Episodic Memory Handler: Adding memories for user {user_id}...")

        for memory in state.extracted_memories:
            if not memory.has_attribute():
                continue

            db_entry = memory.as_db_entry()
            attribute = memory.attribute

            memory_info = db_entry[attribute]
            timestamp = db_entry["timestamp"]

            collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {
                        f"memory.{attribute}": {
                            "information": memory_info,
                            "timestamp": timestamp,
                        }
                    }
                },
                upsert=True,
            )
            logging.info(f"Added to memory.{attribute}: '{memory_info}' at {timestamp}")

    def get_existing_memories(self, user_id):
        query = {"user_id": user_id}
        result = self.memory_db.find_one(self.collection_name, query)

        existing_memories = []

        if result and "memory" in result:
            for attr, memories in result["memory"].items():  # Iterate over attributes
                for memory_entry in memories:  # Iterate over memory list
                    existing_memories.append(
                        EpisodicMemory(
                            information=memory_entry["information"],
                            attribute=attr,
                            timestamp=memory_entry["timestamp"],
                        )
                    )

        return existing_memories
