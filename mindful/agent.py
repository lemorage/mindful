# type: ignore

import json

from typing import List

from mindful.utils import get_api_key
from mindful.llm.openai import OpenAI


# TODO: to be replaced by oop style
class Agent:
    def __init__(self, model: str) -> None:
        self.provider = OpenAI(model=model, api_key=get_api_key("OPENAI_API_KEY"))

    def generate_content(self, prompt: str) -> str:
        """Generates content using selected model provider."""
        # Bug source, to be fixed later!
        # self.provider.complete_chat(prompt)
        return """```json
            {{
            "category": "example",
            "context": "example",
            "keywords": ["a", "b"]
            }}
        ```"""  # Placeholder for actual content generation logic

    def generate_metadata(self, content: str) -> dict:
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
            # Extract the JSON part from the response
            json_str = response.split("```json")[1].split("```")[0].strip()
            metadata = json.loads(json_str)
            return metadata["category"], metadata["context"], metadata["keywords"]
        except (IndexError, ValueError) as e:
            print(f"Error extracting metadata: {e}")
            return "unknown", "unknown", []  # Default return values

    def embed(self, content: str) -> List[float]:
        return [0.0] * 768  # Placeholder for embedding logic
