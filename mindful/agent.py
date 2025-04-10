import json

from typing import List, Tuple, cast

from mindful.utils import get_api_key
from mindful.llm.openai import OpenAI


class Agent:
    def __init__(self, model: str) -> None:
        self.provider = OpenAI(model=model, api_key=get_api_key("OPENAI_API_KEY"))

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

            Output: ```json
            {{
            "category": "",
            "context": "",
            "keywords": []
            }}
            ```
        """
        response = self.generate_content(prompt)

        try:
            metadata = json.loads(response)
            return metadata["category"], metadata["context"], metadata["keywords"]
        except (IndexError, ValueError) as e:
            print(f"Error extracting metadata: {e}")
            return "unknown", "unknown", []  # Default return values

    def embed(self, content: str) -> List[float]:
        return self.provider.get_embedding(content)
