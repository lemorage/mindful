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
from mindful.memory.tape import Tape
from mindful.models import (
    TapeInsight,
    TapeMetadata,
)
from mindful.helpers import pydantic_to_openai_tool
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

    def generate_metadata(self, content: str) -> Tuple[str, str, List[str]]:
        """
        Generate dynamic metadata (category, context, keywords) using LLM tool calling
        via the provider's `complete_chat` method.

        Args:
            content: The text content to analyze.

        Returns:
            A tuple containing (category, context, keywords). Returns ("", "", []) on failure.
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
            return "", "", []

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
            return "", "", []
        except Exception as e:
            logger.exception(f"LLM call failed during metadata generation: {e}", exc_info=True)
            return "", "", []

        # 5. Parse and Validate the Result
        if not parsed_response:
            logger.error("Metadata generation failed: Received empty response from LLM provider.")
            return "", "", []

        tool_calls = parsed_response.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
            logger.warning(f"Metadata generation: LLM response did not contain tool calls. Response: {parsed_response}")
            return "", "", []

        metadata_call = tool_calls[0]
        if (
            metadata_call.get("type") != "function"
            or metadata_call.get("function", {}).get("name") != metadata_tool_name
        ):
            logger.warning(f"Metadata generation: Unexpected tool call received: {metadata_call}")
            return "", "", []

        arguments_str = metadata_call.get("function", {}).get("arguments", "{}")
        try:
            metadata = TapeMetadata.model_validate_json(arguments_str)
            logger.info(f"Metadata generated: Category='{metadata.category}', Keywords={metadata.keywords}")
            return (
                str(metadata.category) if metadata.category else "",
                str(metadata.context) if metadata.context else "",
                metadata.keywords,
            )
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON arguments from tool call: {arguments_str}")
            return "", "", []
        except Exception as e:
            logger.error(f"Failed to parse/validate metadata from tool call arguments: {e}. Data: {arguments_str}")
            return "", "", []

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

    def summarize_content(self, content: str, max_length: int = 150) -> Optional[str]:
        """
        Generate a summary of the given content using the configured provider.

        Args:
            content: The text content to summarize.
            max_length: Approximate desired maximum length of the summary in words.

        Returns:
            The generated summary string, or None on failure.
        """
        logger.debug(f"Requesting summarization for content (first 100 chars): '{content[:100]}...'")

        prompt = f"""Please summarize the following text concisely. Aim for a summary under {max_length} words that captures the main points and key information.

Text to Summarize:
\"\"\"
{content}
\"\"\"

Summary:
"""
        messages: List[ChatMessage] = [{"role": "user", "content": prompt}]

        try:
            # TODO: Consider adding a dedicated summarization_model config later
            parsed_response = self.provider.complete_chat(
                model=self.provider.model,
                messages=messages,
                tools=None,
                tool_choice=None,
                temperature=0.5,
                max_tokens=max_length * 4,  # estimate tokens based on words
            )

            if parsed_response and parsed_response.get("content"):
                summary: str = parsed_response["content"].strip()
                logger.info(f"Summarization successful (length: {len(summary)}).")
                return summary
            else:
                logger.warning(
                    f"Summarization failed: LLM response did not contain content. Response: {parsed_response}"
                )
                return None

        except Exception as e:
            logger.exception(f"LLM call failed during summarization: {e}", exc_info=True)
            return None

    def suggest_tape_refinements(self, target_tape: Tape, neighbor_tapes: List[Tape]) -> Optional[TapeInsight]:
        """Uses LLM to analyze a tape and its neighbors, suggesting refinements."""
        logger.debug(f"Requesting refinement suggestions for tape {target_tape.id}...")

        # 1. Format input for LLM prompt
        target_info = f"Target Tape (ID: {target_tape.id}):\nRole: {target_tape.role}\nCategory: {target_tape.metadata.category}\nContext: {target_tape.metadata.context}\nKeywords: {target_tape.metadata.keywords}\nContent: {target_tape.content[:500]}...\n---"
        neighbors_info = "\n".join(
            [
                f"Neighbor {i + 1} (ID: {n.id}):\nRole: {n.role}\nCategory: {n.metadata.category}\nContext: {n.metadata.context}\nKeywords: {n.metadata.keywords}\nContent: {n.content[:300]}..."
                for i, n in enumerate(neighbor_tapes)
            ]
        )

        prompt = f"""You are a memory refinement AI. Analyze the 'Target Tape' in the context of its nearest 'Neighbor Tapes'.
Based *only* on the provided information, suggest potential refinements by calling the '{TapeInsight.__name__}' tool.

Tasks:
1.  **Redundancy/Archival:** If the Target Tape's core information seems entirely covered or made obsolete by one or more neighbors, suggest archiving it (`should_archive: true`).
2.  **Metadata Improvement:** If the Target's category, context, or keywords could be significantly improved based on the neighbors, provide *only the updated fields* in `updated_metadata`.
3.  **Link Suggestion:** If a strong relationship exists between the Target and a Neighbor *that isn't already linked*, suggest adding it via `new_links`. Format as a JSON object where keys are neighbor Tape IDs and values are short strings describing the relationship (e.g., `{{"neighbor_id_1": "provides example for target", "neighbor_id_2": "contradicts target"}}`). Suggest only strong, specific relationships, not vague 'related'.

{target_info}

{neighbors_info}

Call the '{TapeInsight.__name__}' tool with your suggestions. If no refinements are necessary, call the tool with default values.
"""
        messages: List[ChatMessage] = [{"role": "user", "content": prompt}]

        # 2. Define Tool
        try:
            tool_def: ToolDefinition = {"type": "function", "function": TapeInsight.model_json_schema()}
            tool_def["function"]["name"] = TapeInsight.__name__
        except Exception as e:
            logger.exception("Failed to generate schema for TapeInsight tool.", exc_info=True)
            return None

        # 3. Call LLM
        tool_choice: ToolChoice = {"type": "function", "function": {"name": TapeInsight.__name__}}
        parsed_response: Optional[ParsedResponse] = None
        try:
            parsed_response = self.provider.complete_chat(
                model=self.completion_model,
                messages=messages,
                tools=[tool_def],
                tool_choice=tool_choice,
                temperature=0.2,
            )
        except Exception as e:
            logger.exception(f"LLM call failed during refinement suggestion: {e}", exc_info=True)
            return None

        # 4. Parse and Validate Tool Call
        if not parsed_response:
            return None
        tool_calls = parsed_response.get("tool_calls")
        if not tool_calls:
            logger.warning("Refinement suggestion: No tool call.")
            return None

        try:
            arguments_str = tool_calls[0].get("function", {}).get("arguments", "{}")
            extracted_data = json.loads(arguments_str)
            suggestion = TapeInsight.model_validate(extracted_data)
            logger.info(
                f"Refinement suggestion received for tape {target_tape.id}: Archive={suggestion.should_archive}, MetaUpdate={suggestion.updated_metadata is not None}, Links={len(suggestion.new_links or {})}"
            )
            return suggestion
        except Exception as e:
            logger.error(f"Failed to parse/validate refinement suggestion: {e}. Data: {arguments_str}")
            return None
