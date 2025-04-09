# type: ignore

from typing import Any, Dict, List, Optional
import requests
import json

from mindful.llm.llm_base import LLMBase


class OpenAI(LLMBase):
    """OpenAI provider implementation using direct API calls."""

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def format_request(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Format the request payload for OpenAI chat completions."""
        return {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

    def send_request(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to OpenAI chat completions endpoint."""
        url = f"{self.BASE_URL}/chat/completions"
        response = requests.post(
            url,
            headers=self.headers,
            data=json.dumps(formatted_request),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def parse_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI API response."""
        message = raw_response["choices"][0]["message"]
        return {
            "role": message["role"],
            "content": message["content"],
        }

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get text embedding from OpenAI embedding endpoint."""
        embedding_model = model or self.model
        url = f"{self.BASE_URL}/embeddings"
        payload = {
            "input": text,
            "model": embedding_model,
        }
        response = requests.post(
            url,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
