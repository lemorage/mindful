import logging
import os

import pytest
from pytest_mock import MockerFixture

from mindful.config import (
    MindfulConfig,
    load_mindful_config,
)


def test_load_mindful_config_user_provided(mocker: MockerFixture) -> None:
    """Test that load_mindful_config returns a user-provided MindfulConfig object."""
    # Arrange
    config = MindfulConfig(storage_type="chroma", vector_size=1536)

    # Act
    result = load_mindful_config(config)

    # Assert
    assert result is config
    assert result.storage_type == "chroma"
    assert result.vector_size == 1536


def test_load_mindful_config_environment_variables(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> None:
    """Test that load_mindful_config uses environment variables over defaults."""
    # Arrange
    monkeypatch.setenv("MINDFUL_STORAGE_TYPE", "qdrant")
    monkeypatch.setenv("MINDFUL_QDRANT_URL", "http://test:6333")
    monkeypatch.setenv("MINDFUL_QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("MINDFUL_VECTOR_SIZE", "768")
    spy_logger = mocker.spy(logging.getLogger("mindful"), "debug")

    # Act
    result = load_mindful_config()
    print("ID: ", result)

    # Assert
    assert result.storage_type == "qdrant"
    assert result.qdrant_url == "http://test:6333"
    assert result.vector_size == 768
    assert spy_logger.call_count == 1
    assert "Mindful configuration resolved from environment variables or defaults" in spy_logger.call_args[0][0]


def test_load_mindful_config_defaults(mocker: MockerFixture) -> None:
    """Test that load_mindful_config falls back to defaults when no environment variables are set."""
    # Arrange
    spy_logger = mocker.spy(logging.getLogger("mindful"), "debug")

    # Act
    result = load_mindful_config()

    # Assert
    # Should all be None here but in decorator it will have some defaults after None
    assert result.storage_type == None
    assert result.vector_size == None
    assert result.agent_provider == None
    assert spy_logger.call_count == 1
    assert "Mindful configuration resolved from environment variables or defaults" in spy_logger.call_args[0][0]


def test_load_mindful_config_invalid_env_variable(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> None:
    """Test that load_mindful_config raises RuntimeError for invalid environment variables."""
    # Arrange
    monkeypatch.setenv("MINDFUL_STORAGE_TYPE", "invalid")
    spy_logger = mocker.spy(logging.getLogger("mindful"), "error")

    # Act / Assert
    with pytest.raises(RuntimeError, match="Invalid mindful configuration"):
        load_mindful_config()
    assert spy_logger.call_count == 1
    assert "Failed to resolve configuration" in spy_logger.call_args[0][0]


def test_load_mindful_config_missing_required_field(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> None:
    """Test that load_mindful_config raises RuntimeError for missing required fields (e.g., vector_size for chroma)."""
    # Arrange
    monkeypatch.setenv("MINDFUL_STORAGE_TYPE", "chroma")
    monkeypatch.setenv("MINDFUL_VECTOR_SIZE", "")  # Invalid, should trigger validation error
    spy_logger = mocker.spy(logging.getLogger("mindful"), "error")

    # Act / Assert
    with pytest.raises(RuntimeError, match="Invalid mindful configuration"):
        load_mindful_config()
    assert spy_logger.call_count == 1
    assert "Failed to resolve configuration" in spy_logger.call_args[0][0]


def test_mindful_config_direct_instantiation() -> None:
    """Test MindfulConfig instantiation with direct values."""
    # Arrange
    config_data = {
        "storage_type": "chroma",
        "vector_size": 1536,
        "chroma_collection_name": "test_collection",
        "agent_provider": "openai",
    }

    # Act
    config = MindfulConfig(**config_data)

    # Assert
    assert config.storage_type == "chroma"
    assert config.vector_size == 1536
    assert config.chroma_collection_name == "test_collection"
    assert config.agent_provider == "openai"
    assert config.chroma_path is None  # not provided, should be None


def test_mindful_config_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MindfulConfig instantiation with environment variables."""
    # Arrange
    monkeypatch.setenv("MINDFUL_STORAGE_TYPE", "qdrant")
    monkeypatch.setenv("MINDFUL_VECTOR_SIZE", "768")
    monkeypatch.setenv("MINDFUL_QDRANT_URL", "http://test:6333")
    monkeypatch.setenv("MINDFUL_QDRANT_COLLECTION", "test_qdrant")
    monkeypatch.setenv("MINDFUL_AGENT_PROVIDER", "anthropic")

    # Act
    config = MindfulConfig()
    print("ID: ", config)

    # Assert
    assert config.storage_type == "qdrant"
    assert config.vector_size == 768
    assert config.qdrant_url == "http://test:6333"
    assert config.qdrant_collection_name == "test_qdrant"
    assert config.agent_provider == "anthropic"


def test_mindful_config_defaults() -> None:
    """Test MindfulConfig instantiation with default values."""
    # Arrange
    # No environment variables set

    # Act
    config = MindfulConfig()

    # Assert
    assert config.storage_type is None
    assert config.vector_size is None
    assert config.chroma_path is None
    assert config.chroma_collection_name is None
    assert config.qdrant_url is None
    assert config.qdrant_collection_name is None
    assert config.pinecone_api_key is None
    assert config.pinecone_index_name is None
    assert config.agent_provider is None


def test_mindful_config_ignore_extra_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that MindfulConfig ignores extra environment variables."""
    # Arrange
    monkeypatch.setenv("MINDFUL_STORAGE_TYPE", "chroma")
    monkeypatch.setenv("MINDFUL_VECTOR_SIZE", "1536")
    monkeypatch.setenv("MINDFUL_UNKNOWN_VAR", "something")

    # Act
    config = MindfulConfig()

    # Assert
    assert config.storage_type == "chroma"
    assert config.vector_size == 1536
    # No error should be raised for MINDFUL_UNKNOWN_VAR due to extra="ignore"


def test_mindful_config_invalid_storage_type() -> None:
    """Test that invalid storage_type raises a validation error."""
    # Arrange
    config_data = {"storage_type": "invalid", "vector_size": 1536}

    # Act / Assert
    with pytest.raises(ValueError, match="Input should be 'chroma', 'qdrant' or 'pinecone'"):
        MindfulConfig(**config_data)


def test_validate_storage_requirements_chroma_valid() -> None:
    """Test validate_storage_requirements for valid chroma config."""
    # Arrange
    config = MindfulConfig(storage_type="chroma", vector_size=1536)

    # Act
    result = config.validate_storage_requirements()

    # Assert
    assert result == config
    assert config.storage_type == "chroma"
    assert config.vector_size == 1536


def test_validate_storage_requirements_chroma_missing_vector_size() -> None:
    """Test validate_storage_requirements for chroma with missing vector_size."""
    # Arrange
    config_data = {"storage_type": "chroma"}

    # Act / Assert
    with pytest.raises(ValueError, match="Chroma requires 'vector_size'"):
        MindfulConfig(**config_data)


def test_validate_storage_requirements_qdrant_valid() -> None:
    """Test validate_storage_requirements for valid qdrant config."""
    # Arrange
    config = MindfulConfig(storage_type="qdrant", qdrant_collection_name="test_collection")

    # Act
    result = config.validate_storage_requirements()

    # Assert
    assert result == config
    assert config.storage_type == "qdrant"
    assert config.qdrant_collection_name == "test_collection"


def test_validate_storage_requirements_qdrant_missing_collection() -> None:
    """Test validate_storage_requirements for qdrant with missing collection name."""
    # Arrange
    config_data = {"storage_type": "qdrant"}

    # Act / Assert
    with pytest.raises(ValueError, match="Qdrant requires 'qdrant_collection_name'"):
        MindfulConfig(**config_data)
