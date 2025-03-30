import logging
import sys


class CustomStateGraph:
    def __init__(self, state_cls):
        self.state_cls = state_cls  # The class used for states
        self.nodes = {}
        self.entry_point = None
        self.edges = {}

    def add_node(self, name, function):
        """Adds a processing node to the graph."""
        self.nodes[name] = function

    def set_entry_point(self, name):
        """Sets the entry point for the graph."""
        self.entry_point = name

    def add_conditional_edges(self, node, condition_fn, transitions):
        """Adds conditional edges based on a function applied to the state."""
        self.edges[node] = (condition_fn, transitions)

    def add_edge(self, from_node, to_node):
        """Adds a direct edge from one node to another."""
        self.edges[from_node] = (lambda state: "next", {"next": to_node})

    def run(self, initial_state):
        """Executes the graph starting from the entry point."""
        state = initial_state
        current_node = self.entry_point

        while current_node:
            if current_node not in self.nodes:
                break  # End execution if no node exists

            # Process the current node
            node_fn = self.nodes[current_node]
            state = node_fn(state)

            # Determine the next node
            if current_node in self.edges:
                condition_fn, transitions = self.edges[current_node]
                transition_key = condition_fn(state)

                if transition_key in transitions:
                    current_node = transitions[transition_key]
                else:
                    break  # Stop if no valid transition exists
            else:
                break  # Stop if no edges exist

        return state  # Return the final state

    def run_with_logging(self, initial_state):
        """Executes the graph while logging each step."""
        state = initial_state
        current_node = self.entry_point
        execution_path = []

        logging.info(f"Starting graph execution at node: {current_node}")

        while current_node:
            execution_path.append(current_node)

            # Execute the node
            if current_node not in self.nodes:
                logging.error(f"Node '{current_node}' not found!")
                break

            logging.info(f"Executing node: {current_node}")
            output = self.nodes[current_node](state)
            logging.info(f"Output of {current_node}: {output}")

            # Determine next node
            if current_node in self.edges:
                edge = self.edges[current_node]

                if isinstance(edge, tuple):  # Conditional edge
                    condition_fn, transitions = edge
                    decision = condition_fn(state)
                    next_node = transitions.get(decision, None)
                    logging.info(
                        f"Decision at '{current_node}': {decision} → Next node: {next_node}"
                    )
                else:  # Normal edge
                    next_node = edge
                    logging.info(f"Moving to next node: {next_node}")
            else:
                next_node = None  # End of graph

            current_node = next_node  # Move to the next node

        logging.info(f"Graph execution completed. Path taken: {execution_path}")
