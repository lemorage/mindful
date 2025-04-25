import json
import logging
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)

from mindful.llm.anthropic import Anthropic
from mindful.llm.llm_base import (
    ChatMessage,
    LLMBase,
    ParsedResponse,
    ToolChoice,
    ToolDefinition,
)
from mindful.llm.openai import OpenAI
from mindful.models import (
    TapeMetadata,
    pydantic_to_openai_tool,
)
from mindful.utils import get_api_key

logger = logging.getLogger("mindful")


class MindfulAgent:
    """
    Orchestrates LLM interactions for metadata generation and embedding,
    instantiating its own LLM provider based on configuration.
    """

    # Define default models (can be overridden later)
    DEFAULT_OPENAI_COMPLETION_MODEL = "gpt-4o-mini"
    DEFAULT_ANTHROPIC_COMPLETION_MODEL = "claude-3-haiku-20240307"
    DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

    _PROVIDER_MAP: dict[str, Tuple[Callable[[], LLMBase], str, Optional[str]]] = {
        "openai": (
            lambda: OpenAI(model=MindfulAgent.DEFAULT_OPENAI_COMPLETION_MODEL, api_key=get_api_key("OPENAI_API_KEY")),
            DEFAULT_OPENAI_COMPLETION_MODEL,
            DEFAULT_OPENAI_EMBEDDING_MODEL,
        ),
        "anthropic": (
            # Note: Anthropic doesn't do embeddings via API, TODO: change
            lambda: Anthropic(
                model=MindfulAgent.DEFAULT_ANTHROPIC_COMPLETION_MODEL, api_key=get_api_key("ANTHROPIC_API_KEY")
            ),
            DEFAULT_ANTHROPIC_COMPLETION_MODEL,
            None,  # No default embedding model for Anthropic
        ),
        # Add more providers following the tuple format: (provider_factory, default_completion_model, default_embedding_model)
    }

    def __init__(
        self,
        provider_name: str,  # TODO: Add optional model overrides?
        completion_model_override: Optional[str] = None,
        embedding_model_override: Optional[str] = None,
    ) -> None:
        """
        Initializes the MindfulAgent, creating the specified LLM provider instance.

        Args:
            provider_name (str): Name of the provider (e.g., 'openai', 'anthropic').
            completion_model_override (Optional[str]): Specific model to use for metadata generation.
            embedding_model_override (Optional[str]): Specific model to use for embeddings.
        """
        key = provider_name.strip().lower()
        if key not in self._PROVIDER_MAP:
            raise ValueError(f"Unsupported provider '{provider_name}'. Supported: {list(self._PROVIDER_MAP)}")

        provider_factory, default_chat_model, default_embed_model = self._PROVIDER_MAP[key]

        # Instantiate the provider using the factory
        self.provider: LLMBase = provider_factory()

        # Determine models to use (override > default from map)
        self.completion_model = completion_model_override or default_chat_model
        self.embedding_model = embedding_model_override or default_embed_model

        logger.info(
            f"MindfulAgent initialized with provider '{key}'. "
            f"Completion model: '{self.completion_model}', "
            f"Embedding model: '{self.embedding_model or 'Not Applicable'}'"
        )

    def generate_metadata(self, content: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Generate dynamic metadata (category, context, keywords) using LLM tool calling
        via the provider's `complete_chat` method.

        Args:
            content: The text content to analyze.

        Returns:
            A tuple containing (category, context, keywords). Returns (None, None, []) on failure.
        """
        logger.debug(f"Generating metadata using model '{self.completion_model}' for content: '{content[:100]}...'")

        # 1. Define the Tool based on the Pydantic model
        try:
            metadata_tool_def: ToolDefinition = pydantic_to_openai_tool(
                TapeMetadata, "TapeMetadata", "Extract metadata (category, context, keywords)."
            )
            metadata_tool_name = TapeMetadata.__name__
            metadata_tool_def["function"]["name"] = metadata_tool_name
        except Exception as e:
            logger.exception("Failed to generate JSON schema for TapeMetadata tool.", exc_info=True)
            return None, None, []

        # 2. Create the Prompt/Messages
        prompt = f"""
Analyze the following content and extract the specified metadata. Call the '{metadata_tool_name}'
tool with the extracted information. Be concise and relevant.

Content:
\"\"\"
{content}
\"\"\"
"""
        messages: List[ChatMessage] = [{"role": "user", "content": prompt}]

        # 3. Define Tool Choice
        tool_choice: ToolChoice = {"type": "function", "function": {"name": metadata_tool_name}}

        # 4. Call LLM using provider's complete_chat
        parsed_response: Optional[ParsedResponse] = None
        try:
            parsed_response = self.provider.complete_chat(
                model=self.completion_model,  # pass model override
                messages=messages,
                tools=[metadata_tool_def],
                tool_choice=tool_choice,
                temperature=0.1,  # favor deterministic output
            )
        except NotImplementedError as nie:
            logger.error(f"Tool calling not implemented by provider {type(self.provider).__name__}: {nie}")
            return None, None, []
        except Exception as e:
            logger.exception(f"LLM call failed during metadata generation: {e}", exc_info=True)
            return None, None, []

        # 5. Parse and Validate the Result
        if not parsed_response:
            logger.error("Metadata generation failed: Received empty response from LLM provider.")
            return None, None, []

        tool_calls = parsed_response.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
            logger.warning(f"Metadata generation: LLM response did not contain tool calls. Response: {parsed_response}")
            return None, None, []

        metadata_call = tool_calls[0]
        if (
            metadata_call.get("type") != "function"
            or metadata_call.get("function", {}).get("name") != metadata_tool_name
        ):
            logger.warning(f"Metadata generation: Unexpected tool call received: {metadata_call}")
            return None, None, []

        arguments_str = metadata_call.get("function", {}).get("arguments", "{}")
        try:
            metadata = TapeMetadata.model_validate_json(arguments_str)
            logger.info(f"Metadata generated: Category='{metadata.category}', Keywords={metadata.keywords}")
            return (
                str(metadata.category) if metadata.category else None,
                str(metadata.context) if metadata.context else None,
                metadata.keywords,
            )
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON arguments from tool call: {arguments_str}")
            return None, None, []
        except Exception as e:
            logger.error(f"Failed to parse/validate metadata from tool call arguments: {e}. Data: {arguments_str}")
            return None, None, []

    def embed(self, content: str) -> Optional[List[float]]:
        """
        Generate embeddings for the given content using the configured provider and model.

        Args:
            content: The text content to embed.

        Returns:
            A list of floats representing the embedding, or None if embedding fails or is unsupported.
        """
        if not self.embedding_model:
            logger.warning(f"No embedding model configured for provider {type(self.provider).__name__}. Cannot embed.")
            return None

        if not callable(getattr(self.provider, "get_embedding", None)):
            logger.error(f"Provider {type(self.provider).__name__} does not implement 'get_embedding'.")
            return None

        logger.debug(f"Generating embedding using model '{self.embedding_model}' for content: '{content[:50]}...'")
        try:
            embedding_vector = self.provider.get_embedding(text=content, embedding_model=self.embedding_model)
            if embedding_vector is None or not isinstance(embedding_vector, list):
                logger.error("Embedding generation returned invalid result (None or not a list).")
                return None

            logger.info(f"Embedding generated (vector dim: {len(embedding_vector)}).")
            return embedding_vector
        except NotImplementedError:
            # This handles cases like Anthropic explicitly saying no embeddings
            logger.error(f"Embedding is explicitly not implemented by provider: {type(self.provider).__name__}")
            return None
        except Exception as e:
            # Catch potential API errors etc.
            logger.exception(f"Embedding generation failed: {e}", exc_info=True)
            return None

    def summarize_content(self, content: str) -> Optional[str]:
        """
        Generate a summary of the given content using the configured provider.

        Args:
            content: The text content to summarize.

        Returns:
            A string containing the summary, or None if summarization fails.
        """
        logger.debug(f"Generating summary for content: '{content[:50]}...'")
        return ""  # placeholder for actual summarization logic
