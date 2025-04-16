from functools import wraps
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
)

from mindful.memory.tape import (
    Tape,
    TapeDeck,
)

# Define TypeVar for the return type and ParamSpec for the parameters
# Requires Python 3.10+ for ParamSpec
R = TypeVar("R")
P = ParamSpec("P")


def mindful(input: str) -> Callable[[Callable[P, R]], Callable[P, R]]:  # stick to string type for now
    """
    Decorator to automatically store chat interactions as memory Tapes.

    This decorator can be applied to both instance methods and standalone functions.
    It captures user inputs and function responses, storing them as Tapes in a TapeDeck.
    The TapeDeck is associated with the instance or function, ensuring that interactions
    are preserved across calls.

    Args:
        func (Callable[P, R]): The function or method to be decorated.

    Returns:
        Callable[P, R]: The decorated function or method.

    Raises:
        ValueError: If the function or method has no arguments or only has a `self` argument.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """The wrapper function that executes around the original."""
            # --- Step 1: Initial setup & get user input ---
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind_partial(*args, **kwargs)
                bound_args.apply_defaults()
            except TypeError as e:
                raise

            arguments = bound_args.arguments
            if not arguments:
                raise ValueError("Mindful can't wrap a function with no arguments.")
            elif input not in arguments:
                raise ValueError(f"Expected message input '{input}' not found in function signature.")
            original_user_input = bound_args.arguments[input]

            instance_self: Optional[Any] = None  # To hold the instance 'self'
            is_method = "self" in arguments

            tape_deck: TapeDeck
            if is_method and instance_self is not None:
                if not hasattr(instance_self, "_mindful_core"):
                    # TODO: Example provider - could be configurable
                    instance_self._mindful_core = TapeDeck("openai")
                tape_deck = instance_self._mindful_core
            else:
                # Handle standalone functions - attach TapeDeck to the wrapper itself
                if not hasattr(wrapper, "_mindful_core"):
                    setattr(wrapper, "_mindful_core", TapeDeck("openai"))
                tape_deck = getattr(wrapper, "_mindful_core")

            # --- Step 2: Retrieve PAST memory tapes ---
            retrieved_memory_messages: List[Dict[str, str]] = []
            mindful_user_input: str
            try:
                memory_tapes = tape_deck.retrieve_relevant(original_user_input)
                retrieved_memory_messages = [{"role": t.role, "content": t.content} for t in memory_tapes]
                mindful_memory: Optional[str]
                if not retrieved_memory_messages:
                    mindful_memory = "None"
                else:
                    user_contents = [msg["content"] for msg in retrieved_memory_messages if msg["role"] == "user"]
                    mindful_memory = "\n".join(user_contents) if user_contents else "None"

                mindful_user_input = original_user_input + "<MEMORY>" + mindful_memory + "</MEMORY>"
            except Exception as e:
                print(f"Failed to retrieve memory tapes: {e}")
                mindful_user_input = "<MEMORY>Err</MEMORY>"
                # TODO: should give warnings

            # --- Step 3: Store User Input Tape ---
            try:
                user_tape: Optional[Tape] = tape_deck.add_tape(content=original_user_input, role="user")
            except Exception as e:
                # TODO: should give warnings
                user_tape = None  # Mark as failed

            # --- Step 4: Call Original User Function ---
            response: R
            try:
                # Call the user's original function with modified messages
                bound_args.arguments[input] = mindful_user_input
                response = func(*bound_args.args, **bound_args.kwargs)
            except Exception as e:
                raise e

            # --- Step 5: Store Assistant Response & Link ---
            try:
                # Assuming response is string-like or can be cast
                assistant_content = cast(str, response)
                assistant_tape: Optional[Tape] = tape_deck.add_tape(content=assistant_content, role="assistant")
            except Exception as e:
                assistant_tape = None  # Mark as failed

            # Link user input tape to assistant response tape if both were stored
            if user_tape is not None and assistant_tape is not None:
                try:
                    # Check if IDs exist (they should if tapes were created)
                    if hasattr(user_tape, "id") and hasattr(assistant_tape, "id"):
                        tape_deck.link_tapes(user_tape.id, assistant_tape.id, "response_to")
                    else:
                        print("Could not link tapes: ID attribute missing.")
                except Exception as e:
                    print(f"Failed to link tapes ({user_tape.id} -> {assistant_tape.id}): {e}")

            return response  # Return the original function's response

        return wrapper

    return decorator
