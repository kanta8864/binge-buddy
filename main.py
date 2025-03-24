from binge_buddy.agent_state.states import SemanticAgentState
from binge_buddy.memory import SemanticMemory
from binge_buddy.memory_db import MemoryDB
from binge_buddy.message import UserMessage
from binge_buddy.memory_workflow.semantic_workflow import SemanticWorkflow

def main():
    memory_db = MemoryDB()

    user_id = "kanta"

    collection_name = "semantic_memory"  # Collection name

    query = {"user_id": user_id}  # Query to find the user
    result = memory_db.find_one(collection_name, query)  # Fetch the document

    if result:
        existing_memories = []
        print(
            f"Found existing memories for user: {user_id}, Memories: {result['memory']}"
        )
        for attr, information in result["memory"].items():
            existing_memories.append(
                SemanticMemory(information=information, attribute=attr)
            )
    else:
        existing_memories = []

    semantic_workflow = SemanticWorkflow(memory_db)

    content = """
    I am Kanta, I love furries and a lot of kitty cats. I am interested in watching many cat related
    movies but I ocassionally enjoy some doggy content too.
    """
    message = UserMessage(content=content, user_id=user_id, session_id="123")

    state = SemanticAgentState(
        user_id=user_id,
        existing_memories=existing_memories,
        current_user_message=message,
    )

    # semantic_workflow.run(state)
    # Run with logging
    semantic_workflow.run_with_logging(state)

    # Test db entry
    query = {"user_id": user_id}  # Query to find the user
    result = memory_db.find_one(collection_name, query)  # Fetch the document

    print(f"Current DB entry for user: {result}")

if __name__ == "__main__":
    main()
