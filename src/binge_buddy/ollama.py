import requests
import yaml

from openai import OpenAI

from langchain.llms.base import LLM
from langchain.schema import PromptValue  # Import ChatPromptValue


class OllamaLLM(LLM):  # Inherit from the LLM base class
    # model: str = "llama2:7b"  
    model: str = "deepseek-r1:8b"  
    # model: str = "deepseek-llm:7b"
    # model: str = "llama3-gradient:8b"
    temperature: float = 0.0  # Default temperature

    def _call(self, prompt: str, **kwargs) -> str:
        """
        Call the Ollama API with the given prompt and return the response.
        """
        # Convert ChatPromptValue to a string if necessary
        if isinstance(prompt, PromptValue):
            prompt = str(prompt)  # Convert ChatPromptValue to a string

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,  # Use the stringified prompt
            "stream": False,
            "temperature": self.temperature,
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    @property
    def _llm_type(self) -> str:
        return "ollama"

class OpenAILLM(LLM):
    def _call(self, prompt: str, **kwargs) -> str:
        '''
        Call the OpenAI API with the given prompt and return the response.
        '''
        # Read the config file to extract sensitive information
        with open("config.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
            
        model = OpenAI(api_key=config['openai_api_key'])
        
        response = model.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": prompt
                        },
                        {
                            "role": "user",
                            "content": "Generate your response based on the prompt."
                        }
                    ],
                    model="gpt-4o-mini",
                )
        
        return response.choices[0].message.content
        
    @property
    def _llm_type(self) -> str:
        return "openai"
        