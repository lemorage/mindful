import logging
from typing import Type
from pydantic import BaseModel

from mindful.llm.llm_base import ToolDefinition

logger = logging.getLogger("mindful")


def pydantic_to_openai_tool(model: Type[BaseModel], name: str, description: str) -> ToolDefinition:
    """
    Convert a Pydantic model to an OpenAI function tool definition.
    """
    try:
        schema = model.model_json_schema()
    except Exception as e:
        logger.error(f"Failed to generate schema for {model.__name__}: {e}")
        raise ValueError(f"Invalid schema for {model.__name__}")

    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", []),
            },
        },
    }

    return tool_def
