import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import requests
import yaml
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.schema import PromptValue  # Import ChatPromptValue
from langchain_core.prompt_values import ChatPromptValue
from openai import OpenAI


class OllamaLLM(LLM):  # Inherit from the LLM base class
    # model: str = "llama2:7b"
    model: str = "deepseek-r1:8b"
    # model: Optional[str] = "neural-chat:7b"
    # model: str = "deepseek-llm:7b"
    # model: str = "llama3-gradient:8b"
    temperature: float = 0.0  # Default temperature
    gpu_layers: int = 15
    port: int = 11434  # Default port
    url: str = f"http://localhost:{port}"

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1:8b",
        port: Optional[int] = 11434,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Use instance-level attributes to avoid Pydantic conflicts
        self.model = model or self.model
        self.port = port or self.port
        self.url = f"http://localhost:{self.port}"

        self.check_and_pull_model()

    def is_server_running(self) -> bool:
        """Check if the Ollama server is running on the specified port."""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def check_and_pull_model(self):
        """Check if the specified model exists; if not, attempt to pull it."""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            response.raise_for_status()
            available_models = [model["name"] for model in response.json()["models"]]
            if self.model not in available_models:
                print(f"Model '{self.model}' not found locally. Attempting to pull...")
                result = subprocess.Popen(
                    ["ollama", "pull", self.model],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                for line in result.stdout:
                    print(line.strip())
                result.wait()
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to pull model '{self.model}'. Error: {result.stderr.decode()}"
                    )
                print(f"Model '{self.model}' pulled successfully.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama API: {e}")

    def _call(self, prompt: str, **kwargs) -> str:
        """
        Call the Ollama API with the given prompt and return the response.
        """
        # Convert ChatPromptValue to a string if necessary
        if isinstance(prompt, ChatPromptValue):
            prompt_messages = prompt.to_messages()

        # Convert messages to a formatted string
        formatted_prompt = "\n".join(
            [f"{msg.type.capitalize()}: {msg.content}" for msg in prompt_messages]
        )

        url = f"{self.url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": formatted_prompt,  # Use the stringified prompt
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
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load environment variables from the root `.env`
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        ENV_PATH = BASE_DIR / ".env"

        if not ENV_PATH.exists():
            raise FileNotFoundError(
                "Error: `.env` file not found! Please create one and add the necessary environment variables."
            )

        load_dotenv(ENV_PATH)

        # Store the API key
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to the .env file or set it as an environment variable."
            )

        self.model = model or self.model
        self.temperature = temperature or self.temperature

    def _call(self, prompt) -> str:
        """
        Call the OpenAI API with the given prompt and return the response.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if isinstance(prompt, ChatPromptValue):
            messages = prompt.to_messages()

        openai_messages = [
            {
                "role": "user" if msg.type == "human" else "system",
                "content": msg.content,
            }
            for msg in messages
        ]

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    @property
    def _llm_type(self) -> str:
        return "openai"
