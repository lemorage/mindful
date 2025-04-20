import json
from unittest.mock import (
    ANY,
    MagicMock,
    patch,
)

import pytest

from mindful.agent import MindfulAgent
from mindful.llm.openai import OpenAI
from mindful.llm.anthropic import Anthropic
from mindful.models import TapeMetadata

# Constants for testing
EXPECTED_VECTOR_LENGTH = 768
STATIC_VALUE = 0.1
DEFAULT_TEST_COMPLETION_MODEL = MindfulAgent.DEFAULT_OPENAI_COMPLETION_MODEL
DEFAULT_TEST_EMBEDDING_MODEL = MindfulAgent.DEFAULT_OPENAI_EMBEDDING_MODEL


@pytest.fixture
def mock_metadata_json_string() -> str:
    """Returns the expected JSON string payload within the tool call arguments."""
    return json.dumps({"category": "Tech", "context": "AI in practice", "keywords": ["AI", "machine learning", "LLM"]})


@pytest.fixture
def mock_successful_tool_call_response(mock_metadata_json_string):
    """Simulates ParsedResponse with a successful tool call."""
    return {
        "role": "assistant",
        "content": None,  # Often None when tool call is present
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": TapeMetadata.__name__,  # Match the expected tool name
                    "arguments": mock_metadata_json_string,
                },
            }
        ],
    }


@pytest.fixture
def mock_openai_provider():
    """Mocks the OpenAI provider instance more directly."""
    mock = MagicMock(spec=OpenAI)
    mock.get_embedding.return_value = [STATIC_VALUE] * EXPECTED_VECTOR_LENGTH
    mock.complete_chat.return_value = {"role": "assistant", "content": "Default mock response"}
    return mock


@pytest.fixture
@patch(
    "mindful.agent.TapeMetadata.model_json_schema",
    return_value={
        "title": "TapeMetadata",
        "type": "object",
        "properties": {
            "category": {"title": "Category", "type": "string"},
            "context": {"title": "Context", "type": "string"},
            "keywords": {"title": "Keywords", "type": "array", "items": {"type": "string"}},
        },
        "required": ["category", "context"],
    },
)
@patch("mindful.utils.get_api_key", return_value="fake-key")
@patch("mindful.agent.OpenAI")
def mock_openai_agent_factory(MockOpenAI, mock_get_api_key, mock_schema, mock_openai_provider):
    """
    Factory fixture to create an Agent instance with mocks.
    Mocks the provider instantiation process.
    """
    MockOpenAI.return_value = mock_openai_provider
    agent = MindfulAgent(provider_name="openai")
    assert agent.provider is mock_openai_provider
    assert agent.completion_model == DEFAULT_TEST_COMPLETION_MODEL
    assert agent.embedding_model == DEFAULT_TEST_EMBEDDING_MODEL
    return agent, mock_openai_provider


@pytest.fixture
def mock_openai_agent(mock_openai_agent_factory):
    """Provides the agent and provider mock instance from the factory."""
    return mock_openai_agent_factory


@pytest.mark.unit
def test_generate_metadata_success(mock_openai_agent, mock_successful_tool_call_response):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_provider.complete_chat.return_value = mock_successful_tool_call_response
    test_content = "AI is changing the world."

    # Act
    category, context, keywords = agent.generate_metadata(test_content)

    # Assert
    assert category == "Tech"
    assert context == "AI in practice"
    assert keywords == ["AI", "machine learning", "LLM"]

    mock_provider.complete_chat.assert_called_once_with(
        model=agent.completion_model,  # Check correct model used
        messages=ANY,
        tools=ANY,
        tool_choice=ANY,
        temperature=ANY,
    )

    call_args, call_kwargs = mock_provider.complete_chat.call_args
    assert call_kwargs["model"] == agent.completion_model
    assert isinstance(call_kwargs["messages"], list)
    assert isinstance(call_kwargs["tools"], list)
    assert call_kwargs["tools"][0]["function"]["name"] == TapeMetadata.__name__
    assert call_kwargs["tool_choice"] == {"type": "function", "function": {"name": TapeMetadata.__name__}}


@pytest.mark.unit
def test_generate_metadata_invalid_json_args(mock_openai_agent):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_response_invalid_json = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": TapeMetadata.__name__,
                    "arguments": '{"category": "Test", "context": "Bad JSON,,}',  # Malformed JSON
                },
            }
        ],
    }
    mock_provider.complete_chat.return_value = mock_response_invalid_json

    # Act
    category, context, keywords = agent.generate_metadata("Some input")

    # Assert: Agent should handle parsing error and return defaults (None, None, [])
    assert category is None
    assert context is None
    assert keywords == []
    mock_provider.complete_chat.assert_called_once()


@pytest.mark.unit
def test_generate_metadata_validation_error(mock_openai_agent):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_response_validation_err = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_val_err",
                "type": "function",
                "function": {
                    "name": TapeMetadata.__name__,
                    # Missing 'category' which TapeMetadata might require
                    "arguments": '{"context": "Validation test", "keywords": ["test"]}',
                },
            }
        ],
    }
    mock_provider.complete_chat.return_value = mock_response_validation_err

    # Act
    category, context, keywords = agent.generate_metadata("Validation input")

    # Assert: Agent should handle Pydantic validation error and return defaults
    assert category is None
    assert context is None
    assert keywords == []
    mock_provider.complete_chat.assert_called_once()


@pytest.mark.unit
def test_generate_metadata_no_tool_call(mock_openai_agent):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_response_no_tool = {"role": "assistant", "content": "I cannot fulfill that request."}
    mock_provider.complete_chat.return_value = mock_response_no_tool

    # Act
    category, context, keywords = agent.generate_metadata("Generate metadata please")

    # Assert: Agent should handle missing tool call and return defaults
    assert category is None
    assert context is None
    assert keywords == []
    mock_provider.complete_chat.assert_called_once()


@pytest.mark.unit
def test_agent_embed_with_mocked_provider_returns_fixed_vector(mock_openai_agent):
    # Arrange
    agent, mock_provider = mock_openai_agent
    test_input = "Test content"

    # Act
    result = agent.embed(test_input)

    # Assert
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == EXPECTED_VECTOR_LENGTH, f"Vector length should be {EXPECTED_VECTOR_LENGTH}"
    assert all(
        isinstance(x, float) and x == STATIC_VALUE for x in result
    ), "All elements should be floats equal to STATIC_VALUE"
    # Check that the correct embedding model was passed
    mock_provider.get_embedding.assert_called_once_with(text=test_input, embedding_model=agent.embedding_model)


@pytest.mark.unit
@pytest.mark.parametrize("content", ["", "Another test", " "])
def test_agent_embed_handles_various_inputs(mock_openai_agent, content):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_provider.get_embedding.reset_mock()

    # Act
    result = agent.embed(content)

    # Assert
    assert isinstance(result, list)
    assert len(result) == EXPECTED_VECTOR_LENGTH
    mock_provider.get_embedding.assert_called_once_with(text=content, embedding_model=agent.embedding_model)


@pytest.mark.unit
@patch("mindful.utils.get_api_key", return_value="fake-anthropic-key")
@patch("mindful.agent.Anthropic")
def test_agent_embed_handles_not_implemented(MockAnthropic, mock_get_api_key):
    # Arrange
    mock_anthropic_provider = MagicMock(spec=Anthropic)
    mock_anthropic_provider.get_embedding.side_effect = NotImplementedError
    MockAnthropic.return_value = mock_anthropic_provider

    agent = MindfulAgent(provider_name="anthropic")

    # Act
    result = agent.embed("Some text")

    # Assert
    assert agent.embedding_model is None
    assert result is None
    mock_anthropic_provider.get_embedding.assert_not_called()
