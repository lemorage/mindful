from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

import requests  # type: ignore

from mindful.llm.llm_base import (
    ChatMessage,
    LLMBase,
    ParsedResponse,
    ToolChoice,
    ToolDefinition,
)


class OpenAI(LLMBase):
    """OpenAI provider implementation using direct API calls."""

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)

    def format_request(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: ToolChoice = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": messages, **kwargs}
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
            else:
                payload["tool_choice"] = "auto"
        return payload

    def send_request(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(f"{OpenAI.BASE_URL}/chat/completions", headers=headers, json=formatted_request)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    def parse_response(self, raw_response: Dict[str, Any]) -> ParsedResponse:
        message = raw_response["choices"][0]["message"]
        parsed = {"role": message["role"]}
        if "content" in message and message["content"]:
            parsed["content"] = message["content"]
        if "tool_calls" in message and message["tool_calls"]:
            parsed["tool_calls"] = message["tool_calls"]
        return parsed

    def get_embedding(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"input": text, "model": embedding_model or self.model}
        response = requests.post(f"{OpenAI.BASE_URL}/embeddings", headers=headers, json=payload)
        response.raise_for_status()
        return cast(List[float], response.json()["data"][0]["embedding"])
