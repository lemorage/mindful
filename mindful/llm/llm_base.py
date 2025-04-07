from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBase(ABC):
    """
    Abstract base class for different LLM providers. Handles the formatting of request objects,
    managing the request lifecycle, and parsing into a standard chat completions response format.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the base LLM class.

        Args:
            model (str): The model name or ID to use.
            api_key (Optional[str]): API key for the provider.
            kwargs: Additional configuration for the provider.
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def format_request(self, messages: List[Dict[str, str]], **kwargs) -> Any:  # type: ignore
        """
        Format the messages into the provider-specific request format.

        Args:
            messages (List[Dict[str, str]]): Chat messages in standardized format (e.g. {'role': 'user', 'content': 'Hi'})
            kwargs: Additional parameters for formatting.

        Returns:
            Any: The formatted request object.
        """
        pass

    @abstractmethod
    def send_request(self, formatted_request: Any) -> Any:
        """
        Send the formatted request to the provider's API.

        Args:
            formatted_request (Any): The request object already formatted.

        Returns:
            Any: The raw response from the provider.
        """
        pass

    @abstractmethod
    def parse_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        Parse the raw response into a standardized chat completion format.

        Args:
            raw_response (Any): The response from the provider.

        Returns:
            Dict[str, Any]: Parsed response in standardized format.
        """
        pass

    def complete_chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:  # type: ignore
        """
        Full cycle: formats the request, sends it, and parses the response.

        Args:
            messages (List[Dict[str, str]]): Chat messages in standard format.
            kwargs: Additional parameters (e.g., temperature, max_tokens).

        Returns:
            Dict[str, Any]: Parsed and formatted chat completion.
        """
        formatted = self.format_request(messages, **kwargs)
        raw_response = self.send_request(formatted)
        return self.parse_response(raw_response)
