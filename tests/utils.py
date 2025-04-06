import os
from dotenv import find_dotenv, load_dotenv

from mindful import mindful

from litellm import completion


class LiteLLMClient:
    def __init__(self, model="gpt-4", api_key=None):
        """
        Lightweight interface for generating completions using a language model.

        Args:
            model (str): The model name (default: "gpt-4").
            api_key (str, optional): API key for auth. Falls back to env if not provided.
        """
        self.model = model
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")

    @mindful
    def chat(self, prompt: str) -> str:
        """
        Generate a response from the model given a user prompt.

        Args:
            prompt (str): The user’s input prompt.

        Returns:
            str: Generated model response.
        """
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
        )
        return response.choices[0].message.content


def load_env():
    """Load environment variables from a .env file if present."""
    _ = load_dotenv(find_dotenv())


def get_api_key(key_name: str) -> str:
    """
    Retrieve an API key from the environment, prioritizing .env files.

    Args:
        key_name (str): The name of the environment variable to fetch (e.g., "OPENAI_API_KEY").

    Returns:
        str: The API key if found.

    Raises:
        ValueError: If the API key is not found.
    """
    load_env()
    api_key = os.getenv(key_name)

    if not api_key:
        raise ValueError(f"API key '{key_name}' not found. Please set it in a .env file or as an environment variable.")

    return api_key
