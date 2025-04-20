from functools import wraps
import inspect
import logging
import sys
import threading
from typing import (
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
from mindful.utils import MindfulLogFormatter

logger = logging.getLogger("mindful")

# Flag and lock to ensure default handler setup happens only once per process safely
_mindful_default_handler_configured = False
_mindful_handler_lock = threading.Lock()


# Define TypeVar for the return type and ParamSpec for the parameters
# Requires Python 3.10+ for ParamSpec
R = TypeVar("R")
P = ParamSpec("P")


def mindful(input: str, debug: bool = False) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Factory for the `mindful` decorator. Configures the user input source
    and logging behavior for the 'mindful' package namespace.

    If debug=True, sets the 'mindful' logger level to DEBUG. If no handlers
    are configured for the 'mindful' logger by the application, a default
    StreamHandler using MindfulLogFormatter is added to ensure debug
    output is visible with consistent formatting.

    Args:
        input (str): The parameter name holding user input in the decorated function.
        debug (bool): Enables DEBUG level logging for the 'mindful' namespace
                      and potentially adds a default handler.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: The configured decorator.
    """
    package_logger = logging.getLogger("mindful")
    global _mindful_default_handler_configured

    with _mindful_handler_lock:
        if debug:
            # Set logger level to DEBUG
            current_level = package_logger.getEffectiveLevel()
            if current_level > logging.DEBUG:
                package_logger.setLevel(logging.DEBUG)
                logger.info(f"Mindful package logger level set to DEBUG.")

            # Add default handler if none exists
            if not package_logger.hasHandlers() and not _mindful_default_handler_configured:
                logger.debug("Adding default debug handler for 'mindful' logger.")
                handler = logging.StreamHandler(sys.stderr)
                handler.setLevel(logging.DEBUG)  # Ensure handler emits DEBUG messages
                handler.setFormatter(MindfulLogFormatter("%(message)s"))
                package_logger.addHandler(handler)
                package_logger.propagate = False  # Prevent propagation to root
                _mindful_default_handler_configured = True
            elif package_logger.hasHandlers():
                logger.debug("Existing handlers found for 'mindful' logger.")
        else:
            # Set to INFO when debug=False to avoid DEBUG logs
            package_logger.setLevel(logging.INFO)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """
        Decorator to automatically store chat interactions as memory Tapes, retrieve and apply them.

        This decorator can be applied to both instance methods and standalone functions.
        It captures user inputs and function responses, storing them as Tapes in a TapeDeck.
        The TapeDeck is associated with the instance or function, ensuring that interactions
        are preserved across calls.

        Args:
            func (Callable[P, R]): The function or method to be decorated.

        Returns:
            Callable[P, R]: The decorated function or method on user end.
        """
        wrapper_logger = logging.getLogger("mindful")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """The wrapper function that executes around the original."""
            # Check the debug flag passed to the factory to control emission
            if debug:
                wrapper_logger.debug(f"Entering mindful wrapper for {func.__name__}")

            # --- Step 1: Initial setup & get user input ---
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind_partial(*args, **kwargs)
                bound_args.apply_defaults()
                if debug:
                    arg_reprs = {
                        k: repr(v)[:100] + ("..." if len(repr(v)) > 100 else "")
                        for k, v in bound_args.arguments.items()
                    }
                    wrapper_logger.debug(f"Arguments bound: {arg_reprs}")
            except TypeError as e:
                wrapper_logger.error(f"Failed to bind arguments for {func.__name__}: {e}")
                raise

            arguments = bound_args.arguments
            if not arguments:
                raise ValueError(f"Function {func.__name__} has no arguments; mindful requires at least one.")
            if input not in arguments:
                raise ValueError(f"Expected input parameter '{input}' not found in {func.__name__} signature.")

            original_user_input = arguments[input]

            if debug:
                wrapper_logger.debug(f"Entering mindful wrapper for {func.__name__}")
                arg_reprs = {k: repr(v)[:100] + ("..." if len(repr(v)) > 100 else "") for k, v in arguments.items()}
                wrapper_logger.debug(f"Arguments bound: {arg_reprs}")
                wrapper_logger.debug(f"User input (from '{input}'): '{str(original_user_input)[:100]}...'")

            # --- Step 2: Initialize TapeDeck for state management ---
            def get_tape_deck() -> TapeDeck:
                """Helper to initialize or retrieve TapeDeck for methods or functions."""
                is_method = "self" in arguments
                if is_method:
                    self_instance = arguments["self"]
                    if not hasattr(self_instance, "_mindful_core"):
                        if debug:
                            wrapper_logger.debug(f"Creating TapeDeck for instance {id(self_instance)}")
                        # TODO: Replace with full TapeDeck initialization when ready
                        self_instance._mindful_core = TapeDeck("openai")  # Placeholder
                    return cast(TapeDeck, self_instance._mindful_core)

                # Standalone function
                if not hasattr(wrapper, "_mindful_core"):
                    if debug:
                        wrapper_logger.debug(f"Creating TapeDeck for function {func.__name__}")
                    # TODO: Replace with full TapeDeck initialization when ready
                    setattr(wrapper, "_mindful_core", TapeDeck("openai"))  # Placeholder
                return cast(TapeDeck, getattr(wrapper, "_mindful_core"))

            tape_deck = get_tape_deck()
            if debug:
                wrapper_logger.debug(f"Using TapeDeck instance: {tape_deck!r} (id: {id(tape_deck)})")

            # --- Step 2: Retrieve PAST memory tapes ---
            retrieved_memory_messages: List[Dict[str, str]] = []
            mindful_user_input: str
            if debug:
                wrapper_logger.debug(f"Attempting retrieval for query: '{str(original_user_input)[:100]}...'")
            try:
                memory_tapes = tape_deck.retrieve_relevant(str(original_user_input))
                retrieved_memory_messages = [{"role": t.role, "content": t.content} for t in memory_tapes]
                wrapper_logger.info(f"Retrieved {len(retrieved_memory_messages)} messages from history.")

                if retrieved_memory_messages:
                    memory_log = "\n".join(
                        f"{msg['role'].capitalize()}: {msg['content']}" for msg in retrieved_memory_messages
                    )
                    mindful_user_input = f"<CONTEXT>\n{memory_log}\n</CONTEXT>\nUser: {original_user_input}"
                else:
                    mindful_user_input = f"User: {original_user_input}"

                if debug:
                    wrapper_logger.debug(
                        f"Prepared mindful_user_input (len {len(mindful_user_input)}): '{mindful_user_input[:150]}...'"
                    )
            except Exception as e:
                wrapper_logger.error(f"Failed to retrieve memory tapes: {e}", exc_info=debug)
                mindful_user_input = "User: " + str(original_user_input) + "\n<CONTEXT>No Context Found...</CONTEXT>"

            # --- Step 3: Store User Input Tape ---
            user_tape: Optional[Tape] = None
            if debug:
                wrapper_logger.debug(
                    f"Attempting to store user input tape: role=user, content='{str(original_user_input)[:50]}...'"
                )
            try:
                user_tape = tape_deck.add_tape(content=str(original_user_input), role="user")
                if debug and user_tape:
                    wrapper_logger.debug(f"Stored user tape successfully: id={getattr(user_tape, 'id', 'N/A')}")
            except Exception as e:
                wrapper_logger.error(f"Failed to store user input tape: {e}", exc_info=debug)

            # --- Step 4: Call Original User Function ---
            response: R
            try:
                # Call the user's original function with modified messages
                if debug:
                    wrapper_logger.debug(f"Calling original function {func.__name__} with modified input arg '{input}'")
                # Modify the specific input argument with the <CONTEXT> string
                bound_args.arguments[input] = mindful_user_input
                # Call using the modified bound arguments
                response = func(*bound_args.args, **bound_args.kwargs)
                if debug:
                    wrapper_logger.debug(f"Original function {func.__name__} execution finished.")
            except Exception as e:
                wrapper_logger.error(
                    f"Error during execution of decorated function {func.__name__}: {e}", exc_info=debug
                )
                raise e

            # --- Step 5: Store Assistant Response & Link ---
            assistant_tape: Optional[Tape] = None
            try:
                assistant_content = cast(str, response)
                if debug:
                    wrapper_logger.debug(
                        f"Attempting to store assistant response tape: role=assistant, content='{assistant_content[:50]}...'"
                    )
                assistant_tape = tape_deck.add_tape(content=assistant_content, role="assistant")
                if debug and assistant_tape:
                    wrapper_logger.debug(
                        f"Stored assistant tape successfully: id={getattr(assistant_tape, 'id', 'N/A')}"
                    )
            except Exception as e:
                wrapper_logger.error(f"Failed to store assistant response tape: {e}", exc_info=debug)

            # Link tapes if both were stored successfully
            if user_tape is not None and assistant_tape is not None:
                user_tape_id = getattr(user_tape, "id", None)
                assistant_tape_id = getattr(assistant_tape, "id", None)
                if user_tape_id is not None and assistant_tape_id is not None:
                    if debug:
                        wrapper_logger.debug(
                            f"Attempting to link tapes: user_id={user_tape_id}, assistant_id={assistant_tape_id}"
                        )
                    try:
                        tape_deck.link_tapes(user_tape_id, assistant_tape_id, "response_to")
                        if debug:
                            wrapper_logger.debug("Tapes linked successfully.")
                    except Exception as e:
                        wrapper_logger.error(
                            f"Failed to link tapes ({user_tape_id} -> {assistant_tape_id}): {e}", exc_info=debug
                        )
                elif debug:
                    wrapper_logger.warning("Could not link tapes: ID attribute missing on user or assistant tape.")
            elif debug:
                wrapper_logger.debug("Skipping tape linking as user or assistant tape is missing.")

            if debug:
                wrapper_logger.debug(f"Exiting mindful wrapper for {func.__name__}.")
            return response  # Return the original function's response

        return wrapper

    return decorator
