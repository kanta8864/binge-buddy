import json
from abc import ABC, abstractmethod
from typing import Optional

from langchain.tools import StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from binge_buddy.agent_state.states import AgentState
from binge_buddy.enums import Action, Attribute

# from langgraph.graph import StateGraph
from binge_buddy.state_graph import CustomStateGraph


class MultiAgentWorkflow(ABC):

    state_graph: CustomStateGraph

    def __init__(self): ...

    @abstractmethod
    def run(self, initial_state: AgentState):
        pass
