import time
import pytest

from tests.utils import LiteLLMClient

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def client():
    return LiteLLMClient(model="gpt-3.5-turbo")


def test_one_round_chat_creates_user_and_assistant_tapes(client):
    prompt = "Hello, who are you?"
    response = client.chat(prompt)

    assert isinstance(response, str)
    assert hasattr(client, "_mindful_core")

    tapes = list(client._mindful_core.tapes.values())
    assert len(tapes) >= 2  # Ensure at least 1 user + 1 assistant

    user_tape = next(t for t in tapes if t.content == prompt and t.role == "user")
    assistant_tape = next(t for t in tapes if t.content == response and t.role == "assistant")

    assert user_tape is not None
    assert assistant_tape is not None
    assert assistant_tape.id in user_tape.links or user_tape.id in assistant_tape.links


def test_multiple_interactions_stored_and_linked(client):
    conversation = ["What's the capital of France?", "Who was the first president of the US?"]

    responses = []
    for q in conversation:
        responses.append(client.chat(q))
        time.sleep(1)  # Be kind to the rate limit gods

    tapes = list(client._mindful_core.tapes.values())

    for q, r in zip(conversation, responses):
        user_tape = next((t for t in tapes if t.content == q and t.role == "user"), None)
        assistant_tape = next((t for t in tapes if t.content == r and t.role == "assistant"), None)

        assert user_tape and assistant_tape
        assert assistant_tape.id in user_tape.links or user_tape.id in assistant_tape.links
