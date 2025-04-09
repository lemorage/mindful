# type: ignore

from functools import wraps
from typing import Any, Dict

from mindful.memory.tape import TapeDeck


def mindful(chat_method: Any):
    """
    Decorator to automatically store chat interactions as memory Tapes.
    """

    @wraps(chat_method)
    def wrapper(self, user_input: str, *args: Any, **kwargs: Dict[str, Any]) -> str:
        """This will handle memory operations behind the scenes."""
        if not hasattr(self, "_mindful_core"):
            self._mindful_core = TapeDeck("openai")  # TODO: to make it dynamically set

        # Store user input as a 'user' Tape
        user_tape = self._mindful_core.add_tape(content=user_input, role="user")

        # Generate assistant's response using the original chat method
        response = chat_method(self, user_input, *args, **kwargs)

        # Store assistant's response as an 'assistant' Tape
        assistant_tape = self._mindful_core.add_tape(content=response, role="assistant")

        self._mindful_core.link_tapes(user_tape.id, assistant_tape.id, "response_to")
        return response

    return wrapper
