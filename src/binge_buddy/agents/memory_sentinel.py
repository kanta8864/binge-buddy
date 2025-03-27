import logging

from langchain.llms.base import LLM
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from binge_buddy import utils
from binge_buddy.agent_state.states import AgentState, AgentStateDict
from binge_buddy.agents.base_agent import BaseAgent
from binge_buddy.ollama import OllamaLLM

# Configure logging (if not already configured elsewhere)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MemorySentinel(BaseAgent):
    def __init__(self, llm: LLM):
        super().__init__(
            llm=llm,
            system_prompt_initial="""
            Your job is to assess a brief chat history in order to determine if the conversation contains any details about a user's watching habits regarding streaming content. 
            You are part of a team building a knowledge base regarding a user's watching habits to assist in highly customized streaming content recommendations.
            You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.
             
            You are only interested in the following categories of information:
            1. **Movies and Genres the user likes** 
                - This helps recommend movies the user is likely to enjoy.  
            2. **Movies and Genres the user dislikes**  
                - This helps avoid recommending content the user won't enjoy.  
            3. **Favorite movies**  
                - This helps refine recommendations by finding similar movies.  
            4. **Movies the user wants to watch** 
                - This ensures the system prioritizes unwatched recommendations.  
            5. **Preferred streaming platforms**  
                - This ensures recommendations are available on the user's preferred services.          
            6. **Preferred genres**
                - This helps suggest content that aligns with the user's tastes.
            7. **Personality of the user**  
                - Example: *Enjoys lighthearted comedies; Finds fast-paced movies engaging.*  
            8. **Watching habits** 
                - This helps suggest content that fits the user's lifestyle.  
            9. **Frequency**  
                - This helps suggest content that fits the user's frequency of watching movies/shows.  
            10. **Avoid categories**  
                - Ensures the system respects the user’s hard limits.  
            11. **Character preferences**  
                - Example: *Prefers witty and sarcastic characters; Enjoys dark and mysterious protagonists.*  
            12. **Show length preferences**   
                - This ensures recommendations align with preferred pacing and format.  
            13. **Rewatching tendencies**   
                - This helps tailor recommendations for fresh or nostalgic content.  
            14. **Popularity preferences**  
                - This helps suggest content based on mainstream vs. niche tastes
            
            When you receive a message, you perform a sequence of steps consisting of:
            1. Analyze the message for information.
            2. If it has any information worth recording, return TRUE. If not, return FALSE.
            
            You should ONLY RESPOND WITH TRUE OR FALSE. Absolutely no other information should be provided. Response should only contain one word.
             ### **Response Rules:**  
             If the message contains **any** of the above details → Respond **TRUE**  
             If the message contains **none** of the above → Respond **FALSE**  

            ### **IMPORTANT**:  
             **Even one relevant fact counts as TRUE.**  
             **Do NOT explain your answer, just respond with TRUE or FALSE.**  
            """,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt_initial),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Remember, only respond with TRUE or FALSE. Do not provide any other information.",
                ),
            ]
        )
        self.llm_runnable = RunnableLambda(lambda x: self.llm._call(x))
        self.memory_sentinel_runnable = self.prompt | self.llm_runnable

    def process(self, state: AgentState) -> AgentState:

        # Run the pipeline and get the response
        response = utils.remove_think_tags(
            self.memory_sentinel_runnable.invoke(
                [state.current_user_message.to_langchain_message()]
            )
        )

        # Return True/False based on the response
        state.contains_information = response.lower() == "true"

        logging.info(f"Sentinel Response: {response}")

        return state
