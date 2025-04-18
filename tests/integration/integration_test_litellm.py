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


def test_memory_is_isolated_across_instances():
    client1 = LiteLLMClient(model="gpt-3.5-turbo")
    client1.chat("Hello!")
    client1.chat("How are you?")

    client2 = LiteLLMClient(model="gpt-3.5-turbo")

    assert not hasattr(client2, "_mindful_core")

    client2.chat("What time is it?")

    assert hasattr(client2, "_mindful_core")
    assert len(client2._mindful_core.tapes) == 2  # user + assistant
    assert all("Hello" not in t.content for t in client2._mindful_core.tapes.values())


def test_memory_block_format_is_present(client):
    if hasattr(client, "_mindful_core"):
        client._mindful_core.tapes.clear()

    client.chat("I like spicy food.")
    time.sleep(1)
    response = client.chat("What kind of food do I like?")

    last_user_tape = next(
        t
        for t in reversed(client._mindful_core.tapes.values())
        if t.role == "user" and "What kind of food" in t.content
    )

    assert "spicy" not in last_user_tape.content
    assert "spicy" in response.lower()


def test_memory_injection_enriches_prompt(client):
    """
    Verifies that past interactions are retrieved and injected into subsequent prompts.
    """
    first_prompt = "My favorite color is blue."
    second_prompt = "What did I say my favorite color is?"

    if hasattr(client, "_mindful_core"):
        client._mindful_core.tapes.clear()

    # The memory tape should be stored
    response1 = client.chat(first_prompt)
    assert "blue" in response1.lower()

    # We will use prompt to query the stored memory
    response2 = client.chat(second_prompt)

    # Check if memory was retrieved and affected response
    assert any(
        keyword in response2.lower() for keyword in ["blue", "you said", "your favorite color"]
    ), f"Expected memory-based answer, got: {response2}"

    last_user_tape = next(
        t for t in client._mindful_core.tapes.values() if t.role == "user" and second_prompt in t.content
    )
    assert "blue" not in last_user_tape.content
