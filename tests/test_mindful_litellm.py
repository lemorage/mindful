import pytest
from unittest.mock import patch
from litellm import ModelResponse, Choices, Message
from tests.utils import LiteLLMClient


@pytest.fixture
def chatbot():
    """Fixture to create a new chatbot instance."""
    return LiteLLMClient()


@pytest.fixture
def mock_agent_and_completion():
    with (
        patch("mindful.agent.Agent.generate_metadata") as mock_generate_metadata,
        patch("mindful.agent.Agent.embed") as mock_embedding,
        patch("tests.utils.completion") as mock_completion,
    ):

        mock_generate_metadata.return_value = ("greeting", "general", ["hello", "assistant"])
        mock_embedding.return_value = [0.7] * 768

        def make_model_response(response_text):
            return ModelResponse(
                id="chatcmpl-mock",
                created=1234567890,
                model="gpt-4-0613",
                object="chat.completion",
                system_fingerprint=None,
                service_tier="default",
                choices=[
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message=Message(
                            role="assistant",
                            content=response_text,
                            tool_calls=None,
                            function_call=None,
                            provider_specific_fields={},
                            annotations=[],
                        ),
                    )
                ],
            )

        yield mock_completion, make_model_response


@pytest.mark.unit
def test_single_round_of_conversation(chatbot, mock_agent_and_completion):
    mock_completion, make_model_response = mock_agent_and_completion

    user_input = "Hello, Assistant!"
    mock_response = "Hello! How can I assist you today?"
    mock_completion.return_value = make_model_response(mock_response)

    response = chatbot.chat(user_input)

    assert response == mock_response
    assert hasattr(chatbot, "_mindful_core")
    assert len(chatbot._mindful_core.tapes) == 2
    assert any(t.content == user_input and t.role == "user" for t in chatbot._mindful_core.tapes.values())
    assert any(t.content == mock_response and t.role == "assistant" for t in chatbot._mindful_core.tapes.values())


@pytest.mark.unit
def test_multiple_rounds_of_conversation(chatbot, mock_agent_and_completion):
    mock_completion, make_model_response = mock_agent_and_completion

    inputs_and_responses = [
        ("Hello, Assistant!", "Hello! How can I assist you today?"),
        ("What is the weather like today?", "I can't provide real-time weather info."),
    ]

    mock_completion.side_effect = [make_model_response(resp) for _, resp in inputs_and_responses]

    responses = []
    for user_input, _ in inputs_and_responses:
        response = chatbot.chat(user_input)
        responses.append(response)

    assert len(chatbot._mindful_core.tapes) == 4
    for (user_input, expected_response), response in zip(inputs_and_responses, responses):
        assert response == expected_response
        assert any(t.content == user_input and t.role == "user" for t in chatbot._mindful_core.tapes.values())
        assert any(t.content == response and t.role == "assistant" for t in chatbot._mindful_core.tapes.values())


@pytest.mark.unit
def test_linking_between_tapes(chatbot, mock_agent_and_completion):
    mock_completion, make_model_response = mock_agent_and_completion

    inputs = ["Hello, Assistant!", "What is the weather like today?"]
    responses = ["Hi there!", "I can't give live weather info."]

    mock_completion.side_effect = [make_model_response(r) for r in responses]

    # Act
    for user_input in inputs:
        chatbot.chat(user_input)

    # Assert
    for user_input, response in zip(inputs, responses):
        user_tape = next(
            t for t in chatbot._mindful_core.tapes.values() if t.content == user_input and t.role == "user"
        )
        assistant_tape = next(
            t for t in chatbot._mindful_core.tapes.values() if t.content == response and t.role == "assistant"
        )
        assert user_tape.id in assistant_tape.links
        assert assistant_tape.id in user_tape.links


@pytest.mark.unit
def test_persistence_of_memory(chatbot, mock_agent_and_completion):
    mock_completion, make_model_response = mock_agent_and_completion

    interactions = [
        ("Hello, Assistant!", "Hi there!"),
        ("What is the weather like today?", "I can't give live weather info."),
    ]

    mock_completion.side_effect = [make_model_response(response) for _, response in interactions]

    # Act
    for user_input, _ in interactions:
        chatbot.chat(user_input)

    # Simulate a new instance using the same memory
    new_chatbot = LiteLLMClient()
    new_chatbot._mindful_core = chatbot._mindful_core

    # Assert
    assert len(new_chatbot._mindful_core.tapes) == 4
    for user_input, response in interactions:
        assert any(t.content == user_input and t.role == "user" for t in new_chatbot._mindful_core.tapes.values())
        assert any(t.content == response and t.role == "assistant" for t in new_chatbot._mindful_core.tapes.values())
