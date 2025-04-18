import json
from typing import (
    Callable,
    List,
    Tuple,
    cast,
)

from mindful.llm.openai import OpenAI
from mindful.llm.anthropic import Anthropic
from mindful.utils import get_api_key


class Agent:
    _PROVIDER_MAP: dict[str, Callable[[], OpenAI | Anthropic]] = {
        "openai": lambda: OpenAI(model="gpt-4o-mini", api_key=get_api_key("OPENAI_API_KEY")),
        "anthropic": lambda: Anthropic(model="claude-3-5-sonnet-20241022", api_key=get_api_key("ANTHROPIC_API_KEY")),
        # Add more providers as needed...
    }

    def __init__(self, provider_name: str) -> None:
        key = provider_name.strip().lower()
        if key not in self._PROVIDER_MAP:  # very unlikely to happen, but for safety
            raise ValueError(f"Unsupported provider '{provider_name}'. Supported: {list(self._PROVIDER_MAP)}")

        self.provider = self._PROVIDER_MAP[key]()

    def generate_content(self, prompt: str) -> str:
        """Generates content using selected model provider."""
        messages = [{"role": "user", "content": prompt}]
        response = self.provider.complete_chat(messages)
        return cast(str, response["content"])

    def generate_metadata(self, content: str) -> Tuple[str, str, List[str]]:
        """Generate dynamic metadata based on the given content using the selected model."""
        prompt = f"""
            You are a metadata extraction assistant. Given the content below, generate metadata in JSON format with the following fields: "category" (broad classification), "context" (situational or thematic context), and "keywords" (list of key terms). Keep it concise, accurate, and safe—no harmful or off-topic content.

            Content: {content}

            Output:
            {{
            "category": "",
            "context": "",
            "keywords": []
            }}
        """
        response = self.generate_content(prompt)

        try:
            metadata = json.loads(response)
            return metadata["category"], metadata["context"], metadata["keywords"]
        except (IndexError, ValueError) as e:
            print(f"Error extracting metadata: {e}")
            return "unknown", "unknown", []  # Default return values

    def embed(self, content: str) -> List[float]:
        return self.provider.get_embedding(text=content, embedding_model="text-embedding-3-large")
