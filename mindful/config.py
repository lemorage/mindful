from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class MindfulConfig(BaseModel):
    """
    Configuration object for the Mindful decorator.

    Settings can be provided here, overriding environment variables and defaults.
    """

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
        default=None, description="Collection name for ChromaDB. Env: MINDFUL_CHROMA_COLLECTION"
    )

    # Qdrant Specific
    qdrant_url: Optional[str] = Field(default=None, description="URL for Qdrant instance. Env: MINDFUL_QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(
        default=None, description="API key for Qdrant Cloud (optional). Env: MINDFUL_QDRANT_API_KEY"
    )
    qdrant_collection_name: Optional[str] = Field(
        default=None, description="Collection name for Qdrant. Env: MINDFUL_QDRANT_COLLECTION"
    )

    # Pinecone Specific
    pinecone_api_key: Optional[str] = Field(default=None, description="API key for Pinecone. Env: PINECONE_API_KEY")
    pinecone_index_name: Optional[str] = Field(
        default=None, description="Index name for Pinecone. Env: MINDFUL_PINECONE_INDEX"
    )
    # Add pinecone_environment for older client if needed, or host for newer

    # Agent Configuration
    agent_provider: Optional[Literal["openai", "anthropic"]] = Field(
        default=None, description="LLM provider for the agent ('openai', 'anthropic'). Env: MINDFUL_PROVIDER"
    )
    # Add fields for agent model overrides here?
    # agent_completion_model: Optional[str] = Field(default=None, ...)
    # agent_embedding_model: Optional[str] = Field(default=None, ...)

    class Config:
        validate_assignment = True

    # Helper to get specific storage config dict based on resolved type
    def get_storage_config(self, resolved_storage_type: str, func_name: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {"vector_size": self.vector_size}  # vector size needed by all potentially
        if resolved_storage_type == "chroma":
            config["path"] = self.chroma_path or f"./mindful_db_{func_name}"
            config["collection_name"] = self.chroma_collection_name or f"tapes_{func_name}"
        elif resolved_storage_type == "qdrant":
            config["url"] = self.qdrant_url or "http://localhost:6333"
            config["collection_name"] = self.qdrant_collection_name or f"tapes_{func_name}"
            config["api_key"] = self.qdrant_api_key  # Can be None
        elif resolved_storage_type == "pinecone":
            config["api_key"] = self.pinecone_api_key
            config["index_name"] = (
                self.pinecone_index_name or f"tapes-{func_name.lower().replace('_','')}"
            )  # example default
            if not config["api_key"] or not config["index_name"]:
                raise ValueError("Pinecone config requires 'pinecone_api_key' and 'pinecone_index_name'.")
        # TODO: Add other types...
        return config

    def get_agent_init_kwargs(self, resolved_provider: str) -> Dict[str, Any]:
        kwargs = {"provider_name": resolved_provider}
        # Add model overrides if they exist in the config object
        # if self.agent_completion_model: kwargs["completion_model_override"] = self.agent_completion_model
        # if self.agent_embedding_model: kwargs["embedding_model_override"] = self.agent_embedding_model
        return kwargs
