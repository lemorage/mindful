from functools import wraps
import inspect
import logging
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

# --- Logging Setup ---
# Configure logging for warnings and debug info from the decorator
# Enhanced format: Level (padded), Logger Name, Module:LineNo - Message
LOG_FORMAT = "%(levelname)-8s %(name)s:%(module)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("mindful")

# Note: To see DEBUG messages emitted by the decorator when debug=True,
# the user might need to configure their application's logging level
# for the 'mindful' logger to DEBUG, e.g.:
# logging.getLogger('mindful').setLevel(logging.DEBUG)


# Define TypeVar for the return type and ParamSpec for the parameters
# Requires Python 3.10+ for ParamSpec
R = TypeVar("R")
P = ParamSpec("P")


def mindful(input: str, debug: bool = False) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Factory for the `mindful` decorator. Configures the user input source and logging behavior.

    Args:
        input (str): The parameter name from the decorated function that contains
            the user input to be recorded. This value will be used to extract the
            corresponding argument dynamically at runtime.
        debug (bool): If True, enables DEBUG-level logging for the decorator's operations.
            Defaults to False. Note: The application's logging configuration must allow
            DEBUG messages from the 'mindful' logger for them to appear.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: The configured decorator.
    """

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

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """The wrapper function that executes around the original."""
            if debug:
                logger.debug(f"Entering mindful wrapper for {func.__name__}")
            # --- Step 1: Initial setup & get user input ---
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind_partial(*args, **kwargs)
                bound_args.apply_defaults()
                if debug:
                    arg_reprs = {k: repr(v) for k, v in bound_args.arguments.items()}
                    logger.debug(f"Arguments bound: {arg_reprs}")
            except TypeError as e:
                logger.error(f"Failed to bind arguments for {func.__name__}: {e}")
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

            if debug and original_user_input:
                logger.debug(f"Identified user input string: '{original_user_input[:50]}...'")  # Log truncated input

            if debug and tape_deck:
                logger.debug(f"Using TapeDeck instance: {tape_deck!r} (id: {id(tape_deck)})")

            # --- Step 2: Retrieve PAST memory tapes ---
            retrieved_memory_messages: List[Dict[str, str]] = []
            mindful_user_input: str
            if debug:
                logger.debug(f"Attempting to retrieve relevant memory for query: '{original_user_input[:50]}...'")
            try:
                memory_tapes = tape_deck.retrieve_relevant(original_user_input)
                retrieved_memory_messages = [{"role": t.role, "content": t.content} for t in memory_tapes]
                logger.info(f"Retrieved {len(retrieved_memory_messages)} messages from history.")
                mindful_memory: Optional[str]
                if not retrieved_memory_messages:
                    mindful_memory = "None"
                else:
                    user_contents = [msg["content"] for msg in retrieved_memory_messages if msg["role"] == "user"]
                    mindful_memory = "\n".join(user_contents) if user_contents else "None"

                mindful_user_input = original_user_input + "<MEMORY>" + mindful_memory + "</MEMORY>"
                if debug and mindful_user_input:
                    logger.debug(f"  - Mindful user input is {mindful_user_input}...'")
            except Exception as e:
                logger.error(f"Failed to retrieve memory tapes: {e}", exc_info=debug)  # Add traceback if debug
                mindful_user_input = "<MEMORY>Err</MEMORY>"
                # TODO: should give warnings

            # --- Step 3: Store User Input Tape ---
            user_tape: Optional[Tape] = None
            if debug:
                logger.debug(f"Attempting to store user input tape: role=user, content='{original_user_input[:50]}...'")
            try:
                user_tape = tape_deck.add_tape(content=original_user_input, role="user")
                if debug and user_tape:
                    logger.debug(f"Stored user tape successfully: id={user_tape.id}")
            except Exception as e:
                logger.error(f"Failed to store user input tape: {e}", exc_info=debug)

            # --- Step 4: Call Original User Function ---
            response: R
            try:
                # Call the user's original function with modified messages
                if debug:
                    logger.debug(f"Calling original function: {func.__name__}")
                bound_args.arguments[input] = mindful_user_input
                response = func(*bound_args.args, **bound_args.kwargs)
                if debug:
                    logger.debug(f"Original function {func.__name__} execution finished.")
            except Exception as e:
                logger.error(f"Error during execution of decorated function {func.__name__}: {e}", exc_info=debug)
                raise e

            # --- Step 5: Store Assistant Response & Link ---
            assistant_tape: Optional[Tape] = None
            try:
                assistant_content = cast(str, response)
                if debug:
                    logger.debug(
                        f"Attempting to store assistant response tape: role=assistant, content='{assistant_content[:50]}...'"
                    )
                assistant_tape = tape_deck.add_tape(content=assistant_content, role="assistant")
                if debug and assistant_tape:
                    logger.debug(f"Stored assistant tape successfully: id={assistant_tape.id}")
            except Exception as e:
                logger.error(f"Failed to store assistant response tape: {e}", exc_info=debug)

            # Link user input tape to assistant response tape if both were stored
            if user_tape is not None and assistant_tape is not None:
                if debug:
                    logger.debug(f"Attempting to link tapes: user_id={user_tape.id}, assistant_id={assistant_tape.id}")
                try:
                    if hasattr(user_tape, "id") and hasattr(assistant_tape, "id"):
                        tape_deck.link_tapes(user_tape.id, assistant_tape.id, "response_to")
                        if debug:
                            logger.debug("Tapes linked successfully.")
                    else:
                        logger.warning("Could not link tapes: ID attribute missing.")
                except Exception as e:
                    logger.error(f"Failed to link tapes ({user_tape.id} -> {assistant_tape.id}): {e}", exc_info=debug)
            elif debug:
                logger.debug("Skipping tape linking as user or assistant tape is missing.")

            if debug:
                logger.debug(f"Exiting mindful wrapper for {func.__name__}.")
            return response  # Return the original function's response

        return wrapper

    return decorator
