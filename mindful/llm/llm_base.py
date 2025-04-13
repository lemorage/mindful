from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal

# Define a type alias for the message format for clarity
ChatMessage = Dict[str, str]

# Define a type alias for the tool definition format (inspired by OpenAI)
# Providers might need to adapt this structure internally.
ToolDefinition = Dict[str, Any]
# Example ToolDefinition structure:
# {
#     "type": "function",
#     "function": {
#         "name": "get_current_weather",
#         "description": "Get the current weather in a given location",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The city and state, e.g. San Francisco, CA",
#                 },
#                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#             },
#             "required": ["location"],
#         },
#     },
# }


# Define a type alias for the expected parsed response format
# It should always contain 'role' and either 'content' or 'tool_calls' (or sometimes both)
ParsedResponse = Dict[str, Any]
# Example ParsedResponse (content): {'role': 'assistant', 'content': 'Hello there!'}
# Example ParsedResponse (tool call):
# {
#     'role': 'assistant',
#     'content': None, # Optional: May be None if only tool calls are present
#     'tool_calls': [
#         {
#             'id': 'call_abc123',
#             'type': 'function',
#             'function': {
#                 'name': 'get_current_weather',
#                 'arguments': '{"location": "Boston, MA"}' # Often JSON string
#             }
#         }
#     ]
# }


# Define ToolChoice type (OpenAI function calling schema)
ToolChoice = Union[
    None,  # Provider decides (often 'auto')
    Literal["auto"],  # Provider decides whether to call tools
    Literal["none"],  # Provider must not call tools
    Literal["required"],  # Provider must call one or more tools
    Dict[str, Dict[str, str]],  # Force a specific tool, e.g. {"type": "function", "function": {"name": "my_function"}}
]


class LLMBase(ABC):
    """
    Abstract base class for Large Language Model (LLM) providers.

    Defines a standard interface for chat completions (including tool/function
    calling) and text embeddings. Concrete implementations must provide
    provider-specific logic for formatting requests, sending them, parsing
    responses, and generating embeddings.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the base LLM provider instance.

        Args:
            model (str): The identifier for the specific LLM model to use
                         (e.g., 'gpt-4o', 'claude-3-opus-20240229').
            api_key (Optional[str]): The API key required for authenticating
                                     with the provider's service. Defaults to None.
            **kwargs (Any): Additional provider-specific configuration options
                           (e.g., base_url, project_id, embedding_model).
        """
        if not model:
            raise ValueError("A model identifier must be provided.")
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def format_request(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: ToolChoice = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Formats the input messages and tool definitions into the structure
        expected by the specific LLM provider's API endpoint.

        Args:
            messages (List[ChatMessage]): A list of message dictionaries, each
                following the standard format (e.g., {'role': 'user', 'content': '...'})
                or {'role': 'tool', 'tool_call_id': '...', 'content': '...'}.
            tools (Optional[List[ToolDefinition]]): An optional list of tool
                definitions available for the LLM to call. The structure should
                be adapted to the provider's requirements. Defaults to None.
            tool_choice (ToolChoice): An optional constraint on tool usage
                (e.g., 'none', 'auto', 'required', or specific tool). Provider
                support may vary. Defaults to None (provider default, often 'auto').
            **kwargs (Any): Additional parameters to include in the request payload,
                           such as 'temperature', 'max_tokens', 'top_p', etc.

        Returns:
            Dict[str, Any]: The provider-specific request payload, typically a
                            dictionary ready to be serialized to JSON.
        """
        pass

    @abstractmethod
    def send_request(self, formatted_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the prepared request payload to the LLM provider's API endpoint.

        Handles network communication, authentication headers, and basic
        HTTP error checking (e.g., raising exceptions for 4xx/5xx responses).

        Args:
            formatted_request (Dict[str, Any]): The provider-specific request
                                               payload generated by `format_request`.

        Returns:
            Dict[str, Any]: The raw response payload from the provider, typically
                            parsed from JSON into a dictionary.
        """
        pass

    @abstractmethod
    def parse_response(self, raw_response: Dict[str, Any]) -> ParsedResponse:
        """
        Parses the raw response payload from the provider into a standardized format.

        Extracts the assistant's message content and/or any requested tool calls.

        Args:
            raw_response (Dict[str, Any]): The raw response payload received
                                           from `send_request`.

        Returns:
            ParsedResponse: A dictionary containing the standardized output.
                Expected keys:
                - 'role': Typically 'assistant'.
                - 'content': The textual response from the LLM (str or None).
                - 'tool_calls': A list of requested tool calls (list or None).
                  Each tool call dict should contain 'id', 'type' ('function'),
                  and 'function' (with 'name' and 'arguments' string).
                At least one of 'content' or 'tool_calls' should generally be present
                in a successful response.
        """
        pass

    def complete_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: ToolChoice = None,
        **kwargs: Any,
    ) -> ParsedResponse:
        """
        Executes the full chat completion cycle: formats the request, sends it,
        and parses the response into a standard format.

        This is the primary method for interacting with the LLM's chat capabilities.

        Args:
            messages (List[ChatMessage]): Chat history and current prompt.
            tools (Optional[List[ToolDefinition]]): Available tools. Defaults to None.
            tool_choice (ToolChoice): Tool usage constraint. Defaults to None.
            **kwargs (Any): Additional generation parameters (e.g., temperature).

        Returns:
            ParsedResponse: The standardized parsed response from the LLM,
                            potentially including text content or tool calls.
        """
        formatted_request = self.format_request(messages=messages, tools=tools, tool_choice=tool_choice, **kwargs)
        raw_response = self.send_request(formatted_request)
        return self.parse_response(raw_response)

    @abstractmethod
    def get_embedding(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        """
        Generates a vector embedding for the given text using a specified
        embedding model.

        Args:
            text (str): The input text to embed.
            embedding_model (Optional[str]): The identifier for the embedding
                model to use. If None, the implementation may use a default model
                associated with the provider or configured during initialization.
                Defaults to None.

        Returns:
            List[float]: The vector embedding as a list of floating-point numbers.

        Raises:
            NotImplementedError: If the provider does not support embeddings or
                                 this method is not implemented.
            ValueError: If input is invalid.
            # Other provider-specific exceptions related to API calls.
        """
        pass

    # TODO: add a batch embedding method for efficiency
    # @abstractmethod
    # def get_embeddings(
    #     self,
    #     texts: List[str],
    #     embedding_model: Optional[str] = None
    # ) -> List[List[float]]:
    #     """Generates vector embeddings for a list of texts."""
    #     pass
