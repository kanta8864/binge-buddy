import json
from typing import List, Optional
from binge_buddy import utils
from binge_buddy.memory import Memory
from binge_buddy.memory_db import MemoryDB
from binge_buddy.memory_handler import MemoryHandler
from binge_buddy.message import AgentMessage, Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class ConversationalAgent:
    memories: Optional[List[Memory]] = None

    def __init__(
        self, llm: OllamaLLM, message_log: MessageLog, memory_handler: MemoryHandler
    ):
        """
        Initializes the BingeBuddy conversational agent.

        :param llm: The LLM model to use (e.g., OllamaLLM).
        :param message_log: The message_log that it needs to be observing
        """
        self.llm = llm
        self.message_log = message_log
        self.memory_handler = memory_handler
        # System prompt for the memory sentinel to decide whether to store information
        self.system_prompt_initial = """
            You are a conversational movie and TV show recommendation assistant "Binge Buddy". Your goal is to provide users with natural, engaging, and concise recommendations based on their preferences. Keep responses friendly and to the point—avoid long-winded explanations.
            You only have to introduce yourself once at the beginnig.

            Guidelines:
            Personalized Suggestions: Ask clarifying questions if needed to tailor recommendations.
            Concise Responses: Keep answers short but informative, focusing on why a show or movie fits the user’s taste.
            Natural Conversation: Respond casually and naturally, like a movie-savvy friend.
            Diverse Picks: Offer a mix of well-known and hidden gems, ensuring variety.
            No Spoilers: Avoid revealing major plot points unless explicitly asked.
            If the user is unsure what to watch, guide them with simple questions (e.g., "Do you want something lighthearted or intense?"). If they ask for specific genres, moods, or themes, match them accordingly.

            Message Logs for context:
            {message_logs}

            Exisiting memory:
            {memories}

            Current message to respond to (Only write respond to this message):
            {message}

            

            Your goal is to make discovering movies and shows fun and effortless! Do not ask too many questions and suggest movies where possible.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="message"),
                MessagesPlaceholder(variable_name="memories"),
                MessagesPlaceholder(variable_name="message_logs"),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))

        self.conversational_agent_runnable = self.prompt | self.llm_runnable

    def process_message(self, message: Message) -> str:
        """
        Analyzes the current message and provide a response.
        """
        existing_memories = self.memory_handler.get_existing_memories(message.user_id)
        print(existing_memories)
        if existing_memories:
            memories = [
                AIMessage(
                    content=json.dumps(
                        [memory.as_dict() for memory in existing_memories]
                    )
                )
            ]

        else:
            memories = []

        # Run the pipeline and get the response
        response = utils.remove_think_tags(
            self.conversational_agent_runnable.invoke(
                {
                    "message": [message.to_langchain_message()],
                    "message_logs": list(
                        map(
                            lambda message: message.to_langchain_message(),
                            self.message_log,
                        )
                    ),
                    "memories": memories,
                }
            )
        )
        self.message_log.add_message(message)

        agent_message = AgentMessage(
            content=response, user_id="user", session_id="session"
        )
        self.message_log.add_message(agent_message)
        return response

    def add_user_message(self, message: Message):
        self.message_log.add_message(message)

    def fetch_user_memories(self, user_id: str):
        return self.memoryDB.get_memories(user_id)


if __name__ == "__main__":
    # Initialize the message log and LLM (for now, using a mock LLM)
    llm = OllamaLLM()
    message_log = MessageLog("user", "session")
    memory_db = MemoryDB()
    semantic_agent = ConversationalAgent(
        llm=llm, message_log=message_log, memory_db=memory_db
    )

    # Test message
    current_message = Message(
        content="I love watching sci-fi movies like The Matrix!",
        role="user",
        session_id="session",
        user_id="user",
    )

    response = semantic_agent.process_message(current_message)
