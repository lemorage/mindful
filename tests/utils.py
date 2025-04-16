from litellm import completion

from mindful import mindful
from mindful.utils import get_api_key


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

    @mindful(input="prompt")
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
