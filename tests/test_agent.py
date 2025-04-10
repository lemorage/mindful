import pytest
from unittest.mock import patch, MagicMock
from mindful.agent import Agent

EXPECTED_VECTOR_LENGTH = 768
STATIC_VALUE = 0.1


@pytest.fixture
def mock_openai_response():
    return {
        "content": '{"category": "Tech", "context": "AI in practice", "keywords": ["AI", "machine learning", "LLM"]}'
    }


@pytest.fixture
def mock_openai_agent():
    """Fixture to create an Agent instance with a mocked provider."""
    mock_provider = MagicMock()
    mock_provider.get_embedding.return_value = [STATIC_VALUE] * EXPECTED_VECTOR_LENGTH
    with patch("mindful.utils.get_api_key", return_value="fake-key"):
        with patch("mindful.agent.OpenAI", return_value=mock_provider):
            agent = Agent(model="gpt-4")
    return agent, mock_provider


@pytest.mark.unit
def test_generate_content(mock_openai_agent, mock_openai_response):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_provider.complete_chat.return_value = mock_openai_response

    # Act
    result = agent.generate_content("Say something smart.")

    # Assert
    assert result == mock_openai_response["content"]
    mock_provider.complete_chat.assert_called_once()


@pytest.mark.unit
def test_generate_metadata(mock_openai_agent, mock_openai_response):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_provider.complete_chat.return_value = mock_openai_response

    # Act
    category, context, keywords = agent.generate_metadata("AI is changing the world.")

    # Assert
    assert category == "Tech"
    assert context == "AI in practice"
    assert keywords == ["AI", "machine learning", "LLM"]


@pytest.mark.unit
def test_generate_metadata_with_invalid_json(mock_openai_agent):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_provider.complete_chat.return_value = {"content": "Not a JSON string"}

    # Act
    category, context, keywords = agent.generate_metadata("Some input")

    # Assert
    assert category == "unknown"
    assert context == "unknown"
    assert keywords == []


@pytest.mark.unit
def test_generate_metadata_parses_json_correctly(mock_openai_agent):
    # Arrange
    agent, mock_provider = mock_openai_agent
    mock_provider.complete_chat.return_value = {
        "content": '{"category": "Science", "context": "Physics", "keywords": ["quantum", "mechanics"]}'
    }

    # Act
    category, context, keywords = agent.generate_metadata("Physics content")

    # Assert
    assert category == "Science"
    assert context == "Physics"
    assert keywords == ["quantum", "mechanics"]


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
    assert all(isinstance(x, float) and x == STATIC_VALUE for x in result), "All elements should be floats equal to 0.1"
    mock_provider.get_embedding.assert_called_once_with(test_input)


@pytest.mark.unit
@pytest.mark.parametrize("content", ["", "Another test", " "])
def test_agent_embed_handles_various_inputs(mock_openai_agent, content):
    # Arrange
    agent, mock_provider = mock_openai_agent

    # Act
    result = agent.embed(content)

    # Assert
    assert isinstance(result, list)
    assert len(result) == EXPECTED_VECTOR_LENGTH
    mock_provider.get_embedding.assert_called_once_with(content)
