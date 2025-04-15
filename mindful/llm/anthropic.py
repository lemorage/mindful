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


class Anthropic(LLMBase):
    """Anthropic provider implementation using direct API calls."""

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(model, api_key, **kwargs)

    def format_request(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: ToolChoice = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Validate and adapt OpenAI-style ToolDefinition to Anthropic's format
        adapted_tools: Optional[List[Dict[str, Any]]] = None
        if tools:
            adapted_tools = []
            for t in tools:
                name = t["function"]["name"]
                if (
                    not name
                    or not isinstance(name, str)
                    or len(name) > 64
                    or not all(c.isalnum() or c in ["_", "-"] for c in name)
                ):
                    raise ValueError(f"Invalid tool name: {name}. Must match ^[a-zA-Z0-9_-]{{1,64}}$")
                description = t["function"]["description"]

                adapted_tools.append(
                    {"name": name, "description": description, "input_schema": t["function"]["parameters"]}
                )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            **kwargs,
        }

        if adapted_tools:
            payload["tools"] = adapted_tools
            # Handle tool_choice per Anthropic's options: auto, any, tool, none
            if tool_choice == "none":
                payload.pop("tools")
            elif tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "any":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice == "required":
                # Map 'required' to Anthropic's 'any' (must use at least one tool)
                payload["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                payload["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}

            # Support disabling parallel tool use
            if kwargs.get("disable_parallel_tool_use", False):
                payload["tool_choice"] = payload.get("tool_choice", {})
                payload["tool_choice"]["disable_parallel_tool_use"] = True

        return payload

    def send_request(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        response = requests.post(f"{Anthropic.BASE_URL}/messages", headers=headers, json=formatted_request)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    def parse_response(self, raw_response: Dict[str, Any]) -> ParsedResponse:
        content: List[Dict[str, Any]] = raw_response.get("content", [])
        stop_reason: Optional[str] = raw_response.get("stop_reason")
        parsed: ParsedResponse = {"role": "assistant"}
        text_content: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in content:
            if block["type"] == "text":
                # Handle chain-of-thought or regular text
                text_content.append(str(block["text"]))
            elif block["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                    }
                )

        if text_content:
            parsed["content"] = "\n".join(text_content)
        if tool_calls:
            parsed["tool_calls"] = tool_calls
        if stop_reason:
            parsed["stop_reason"] = stop_reason  # include for tool_use detection

        return parsed

    def get_embedding(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        raise NotImplementedError("Anthropic does not provide an embedding endpoint as of now.")
