# type: ignore

import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

import requests  # type: ignore[import-untyped]

from mindful.llm.llm_base import LLMBase


class Anthropic(LLMBase):
    """Anthropic provider implementation using direct API calls."""

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",  # Latest stable version as per docs
        }

    def format_request(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Format the request payload for Anthropic messages endpoint."""
        # Anthropic expects a single system message at the start if present
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        payload = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": 1024,  # Required parameter
            **kwargs,
        }
        if system_message:
            payload["system"] = system_message
        return payload

    def send_request(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to Anthropic messages endpoint."""
        url = f"{self.BASE_URL}/messages"
        response = requests.post(
            url,
            headers=self.headers,
            data=json.dumps(formatted_request),
            timeout=30,
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    def parse_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Anthropic API response."""
        # Anthropic returns content as a list of content blocks
        content = "".join(block["text"] for block in raw_response["content"] if block["type"] == "text")
        return {
            "role": raw_response["role"],
            "content": content,
        }

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Anthropic doesn't provide embedding endpoint; raising NotImplementedError."""
        raise NotImplementedError("Anthropic API does not support direct embedding generation")
