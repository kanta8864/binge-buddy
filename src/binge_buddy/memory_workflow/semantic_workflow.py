# from langgraph.graph import END, StateGraph
from binge_buddy.agent_state.states import AgentState, SemanticAgentState
from binge_buddy.agents.aggregator_reviewer import AggregatorReviewer
from binge_buddy.agents.extractor_reviewer import ExtractorReviewer
from binge_buddy.agents.memory_aggregator import MemoryAggregator
from binge_buddy.agents.memory_attributor import MemoryAttributor
from binge_buddy.agents.memory_extractor import MemoryExtractor
from binge_buddy.agents.memory_sentinel import MemorySentinel
from binge_buddy.memory import Memory, SemanticMemory
from binge_buddy.memory_db import MemoryDB
from binge_buddy.memory_handler import SemanticMemoryHandler
from binge_buddy.memory_workflow.multi_agent_workflow import MultiAgentWorkflow
from binge_buddy.message import UserMessage
from binge_buddy.ollama import OllamaLLM
from binge_buddy.state_graph import CustomStateGraph


class SemanticWorkflow(MultiAgentWorkflow):
    def __init__(self, memory_handler):
        super().__init__()

        # todo: change this to use global llm
        llm = OllamaLLM()
        memory_sentinel = MemorySentinel(llm)
        memory_extractor = MemoryExtractor(llm)
        extractor_reviewer = ExtractorReviewer(llm)
        memory_attributor = MemoryAttributor(llm)
        memory_aggregator = MemoryAggregator(llm)
        aggregator_reviewer = AggregatorReviewer(llm)

        memory_handler = memory_handler

        # Initialize a new graph
        self.state_graph: CustomStateGraph = CustomStateGraph(SemanticAgentState)

        # Add Nodes
        self.state_graph.add_node("sentinel", memory_sentinel.process)
        self.state_graph.add_node("memory_extractor", memory_extractor.process)
        self.state_graph.add_node("memory_reviewer", extractor_reviewer.process)
        self.state_graph.add_node("memory_attributor", memory_attributor.process)
        self.state_graph.add_node("memory_aggregator", memory_aggregator.process)
        self.state_graph.add_node("aggregator_reviewer", aggregator_reviewer.process)
        self.state_graph.add_node(
            "memory_handler", memory_handler.process
        )  # Final action node

        # Set the Starting Edge
        self.state_graph.set_entry_point("sentinel")

        # Add Conditional Edges (Updated to use attributes instead of dictionary keys)
        self.state_graph.add_conditional_edges(
            "sentinel",
            lambda state: "yes" if state.contains_information else "no",
            {
                "yes": "memory_extractor",
                "no": None,  # END is typically None in a custom graph
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_extractor",
            lambda state: "continue" if state.extracted_memories else "end",
            {
                "continue": "memory_reviewer",
                "end": None,
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_reviewer",
            lambda state: ("continue" if not state.needs_repair else "repair"),
            {
                "continue": "memory_attributor",
                "repair": "memory_extractor",
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_attributor",
            lambda state: (
                "continue"
                if all(mem.has_attribute() for mem in state.extracted_memories)
                else "end"
            ),
            {
                "continue": "memory_aggregator",
                "end": None,
            },
        )

        self.state_graph.add_conditional_edges(
            "memory_aggregator",
            lambda state: "continue" if state.aggregated_memories else "end",
            {
                "continue": "aggregator_reviewer",
                "end": None,
            },
        )

        self.state_graph.add_conditional_edges(
            "aggregator_reviewer",
            lambda state: ("continue" if not state.needs_repair else "repair"),
            {
                "continue": "memory_handler",
                "repair": "memory_aggregator",
            },
        )

        # Add Normal Edges
        self.state_graph.add_edge("memory_handler", None)

    def run(self, initial_state: AgentState):
        self.state_graph.run(initial_state)

    def run_with_logging(self, initial_state: AgentState):
        self.state_graph.run_with_logging(initial_state)

    def print_nodes(self):
        if self.state_graph is None:
            print("Graph has not been initialized yet.")
            return

        nodes = self.state_graph.nodes
        print("Nodes in the graph:")
        for node in nodes:
            print(f"- {node}")


if __name__ == "__main__":

    memory_db = MemoryDB()
    memory_handler = SemanticMemoryHandler(memory_db)

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

    semantic_workflow = SemanticWorkflow(memory_handler)

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
