import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
)

from pydantic import (
    Field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from mindful.cli import get_mindful_config_dir

logger = logging.getLogger("mindful")


class MindfulConfig(BaseSettings):  # type: ignore
    """
    Configuration object for the Mindful decorator.

    Settings can be provided here, overriding environment variables and defaults.
    """

    model_config = SettingsConfigDict(env_prefix="MINDFUL_", extra="ignore", populate_by_name=True)

    # Storage Configuration
    storage_type: Optional[Literal["chroma", "qdrant", "pinecone"]] = Field(
        default=None, description="Type of vector store ('chroma', 'qdrant', 'pinecone'). Env: MINDFUL_STORAGE_TYPE"
    )
    vector_size: Optional[int] = Field(
        default=None, description="Dimension of embedding vectors. Critical. Env: MINDFUL_VECTOR_SIZE"
    )  # TODO: Ideally infer this from embedding model automatically

    # Chroma Specific
    chroma_path: Optional[str] = Field(
        default=None, description="Path for ChromaDB persistent storage. Env: MINDFUL_CHROMA_PATH"
    )
    chroma_collection_name: Optional[str] = Field(
        default=None,
        alias="MINDFUL_CHROMA_COLLECTION",
        description="Collection name for ChromaDB. Env: MINDFUL_CHROMA_COLLECTION",
    )

    # Qdrant Specific
    qdrant_url: Optional[str] = Field(default=None, description="URL for Qdrant instance. Env: MINDFUL_QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(
        default=None, description="API key for Qdrant Cloud (optional). Env: MINDFUL_QDRANT_API_KEY"
    )
    qdrant_collection_name: Optional[str] = Field(
        default=None,
        alias="MINDFUL_QDRANT_COLLECTION",
        description="Collection name for Qdrant. Env: MINDFUL_QDRANT_COLLECTION",
    )

    # Pinecone Specific
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key. Env: PINECONE_API_KEY")
    pinecone_index_name: Optional[str] = Field(
        default=None, description="Index name for Pinecone. Env: MINDFUL_PINECONE_INDEX"
    )
    # Add pinecone_environment for older client if needed, or host for newer

    # Agent Configuration
    agent_provider: Optional[Literal["openai", "anthropic"]] = Field(
        default=None, description="LLM provider for the agent ('openai', 'anthropic'). Env: MINDFUL_AGENT_PROVIDER"
    )
    # Add fields for agent model overrides here?
    # agent_completion_model: Optional[str] = Field(default=None, ...)
    # agent_embedding_model: Optional[str] = Field(default=None, ...)

    @model_validator(mode="after")
    def validate_storage_requirements(self) -> "MindfulConfig":
        """Ensure required fields are present based on the selected storage type."""
        if self.storage_type == "pinecone":
            if not self.pinecone_api_key:
                raise ValueError("Pinecone requires 'pinecone_api_key'.")
            if not self.pinecone_index_name:
                raise ValueError("Pinecone requires 'pinecone_index_name'.")
        elif self.storage_type == "qdrant":
            if not self.qdrant_collection_name:
                raise ValueError("Qdrant requires 'qdrant_collection_name'.")
        elif self.storage_type == "chroma":
            if not self.vector_size:
                raise ValueError("Chroma requires 'vector_size'.")
        return self

    def get_storage_config(self, resolved_storage_type: str, func_name: str) -> Dict[str, Any]:
        """
        Generate storage configuration dictionary for the given storage type.

        Args:
            resolved_storage_type (str): The resolved storage type (e.g., 'chroma', 'qdrant').
            func_name (str): The name of the decorated function for default naming.

        Returns:
            Dict[str, Any]: Configuration dictionary for the storage backend.
        """
        config_dir = get_mindful_config_dir()
        config: Dict[str, Any] = {"vector_size": self.vector_size}  # vector size needed by all potentially
        if resolved_storage_type == "chroma":
            default_path = str(config_dir / "db" / f"mindful_db_{func_name}")
            config["path"] = self.chroma_path or default_path
            config["collection_name"] = self.chroma_collection_name or f"tapes_{func_name}"
            Path(config["path"]).parent.mkdir(parents=True, exist_ok=True)
        elif resolved_storage_type == "qdrant":
            config["url"] = self.qdrant_url or "http://localhost:6333"
            config["collection_name"] = self.qdrant_collection_name or f"tapes_{func_name}"
            config["api_key"] = self.qdrant_api_key  # can be None
        elif resolved_storage_type == "pinecone":
            config["api_key"] = self.pinecone_api_key
            config["index_name"] = (
                self.pinecone_index_name or f"tapes-{func_name.lower().replace('_', '')}"
            )  # example default
            if not config["api_key"] or not config["index_name"]:
                raise ValueError("Pinecone config requires 'pinecone_api_key' and 'pinecone_index_name'.")
        else:
            raise ValueError(f"Unsupported storage type: {resolved_storage_type}")
        # TODO: Add other types...
        return config

    def get_agent_init_kwargs(self, resolved_provider: str) -> Dict[str, Any]:
        """
        Generate initialization kwargs for the MindfulAgent.

        Args:
            resolved_provider (str): The resolved agent provider (e.g., 'openai', 'anthropic').

        Returns:
            Dict[str, Any]: Keyword arguments for agent initialization.
        """
        kwargs = {"provider_name": resolved_provider}
        # Add model overrides if they exist in the config object
        # if self.agent_completion_model: kwargs["completion_model_override"] = self.agent_completion_model
        # if self.agent_embedding_model: kwargs["embedding_model_override"] = self.agent_embedding_model
        return kwargs


def load_mindful_config(config: Optional[MindfulConfig] = None) -> MindfulConfig:
    """
    Load the Mindful configuration with the following precedence:
    1. User-provided MindfulConfig object (if provided).
    2. Environment variables (via Pydantic BaseSettings).
    3. Defaults defined in MindfulConfig.

    Args:
        config (Optional[MindfulConfig]): User-provided configuration object.

    Returns:
        MindfulConfig: The resolved configuration object.
    """
    if config is not None:
        return config

    try:
        resolved_config = MindfulConfig()
        logger.debug("Mindful configuration resolved from environment variables or defaults.")
        return resolved_config
    except Exception as e:
        logger.error(f"Failed to resolve configuration: {e}")
        raise RuntimeError(f"Invalid mindful configuration: {e}")
