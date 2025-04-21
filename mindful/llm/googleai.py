import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

import requests  # type: ignore[import-untyped]

from mindful.llm.llm_base import (
    ChatMessage,
    LLMBase,
    ParsedResponse,
    ToolChoice,
    ToolDefinition,
)


class GoogleAI(LLMBase):
    """GoogleAI provider implementation using direct API calls."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        self.project_id = kwargs.get("project_id")

    def format_request(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: ToolChoice = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Convert messages to Google AI format
        contents = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages]

        payload = {"contents": contents, **kwargs}
        if tools:
            # Adapt ToolDefinition to Google AI's function declarations
            function_declarations = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "parameters": t["function"]["parameters"],
                }
                for t in tools
            ]
            payload["tools"] = [{"function_declarations": function_declarations}]
            if isinstance(tool_choice, dict):
                payload["tool_config"] = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool_choice["function"]["name"]],
                    }
                }
            elif tool_choice == "none":
                payload.pop("tools")
        return payload

    def send_request(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        url = f"{GoogleAI.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"
        if self.project_id:
            url += f"&alt_project={self.project_id}"

        response = requests.post(url, headers=headers, json=formatted_request)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    def parse_response(self, raw_response: Dict[str, Any]) -> ParsedResponse:
        candidate = raw_response["candidates"][0]["content"]
        parsed = {"role": candidate["role"]}
        text_content = None
        tool_calls = []

        for part in candidate["parts"]:
            if "text" in part:
                text_content = part["text"]
            elif "functionCall" in part:
                tool_calls.append(
                    {
                        "id": f"call_{id(part)}",  # Google AI doesn't provide IDs; generate one
                        "type": "function",
                        "function": {
                            "name": part["functionCall"]["name"],
                            "arguments": json.dumps(part["functionCall"]["args"]),
                        },
                    }
                )

        if text_content:
            parsed["content"] = text_content
        if tool_calls:
            parsed["tool_calls"] = tool_calls
        return parsed

    def get_embedding(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        headers = {"Content-Type": "application/json"}
        model = embedding_model or "textembedding-gecko"
        url = f"{GoogleAI.BASE_URL}/models/{model}:embedContent?key={self.api_key}"
        payload = {"content": {"parts": [{"text": text}]}}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return cast(List[float], response.json()["embedding"]["value"])
