from functools import wraps
import inspect
from typing import (
    Any,
    Callable,
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
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            arguments = bound_args.arguments
            if not arguments:
                raise ValueError("Mindful can't wrap a function with no arguments.")
            elif input not in arguments:
                raise ValueError(f"Expected message input '{input}' not found in function signature.")
            original_input = bound_args.arguments[input]

            is_method = "self" in arguments

            tape_deck: TapeDeck

            # Get or create TapeDeck
            if is_method:
                # 'self' here refers to the instance the method is bound to.
                # Mypy might complain here too if the class definition doesn't
                # hint at _mindful_core. Using Any or Protocols on the class
                # side might be needed, or ignoring these lines as well.
                instance_self: Any = arguments["self"]  # Use Any for now if class type is unknown
                if not hasattr(instance_self, "_mindful_core"):
                    instance_self._mindful_core = TapeDeck("openai")
                tape_deck = instance_self._mindful_core
            else:
                # Handle dynamically added attribute to the wrapper function object
                if not hasattr(wrapper, "_mindful_core"):
                    # Tell mypy to ignore the dynamic attribute assignment
                    setattr(wrapper, "_mindful_core", TapeDeck("openai"))  # Use setattr for clarity
                    # Alternative:
                    # wrapper._mindful_core = TapeDeck("openai") # type: ignore[attr-defined]
                # Tell mypy to ignore the dynamic attribute access
                tape_deck = getattr(wrapper, "_mindful_core")  # Use getattr for clarity
                # Alternative:
                # tape_deck = wrapper._mindful_core # type: ignore[attr-defined]

            # Store user input
            user_tape: Tape = tape_deck.add_tape(content=original_input, role="user")

            # Call original function
            response = func(*args, **kwargs)

            # Store assistant response
            assistant_tape: Tape = tape_deck.add_tape(content=cast(str, response), role="assistant")

            if hasattr(user_tape, "id") and hasattr(assistant_tape, "id"):
                tape_deck.link_tapes(user_tape.id, assistant_tape.id, "response_to")
            else:
                print("Warning: Could not link tapes, ID attribute missing.")

            return response

        return wrapper

    return decorator
