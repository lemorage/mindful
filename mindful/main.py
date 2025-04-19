import ast
from functools import wraps
import inspect
import logging
from logging import LogRecord
import re
import time
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
_mindful_logger = logging.getLogger("Mindful")
_mindful_logger_initialized = False  # Only set this once per process

# Define TypeVar for the return type and ParamSpec for the parameters
# Requires Python 3.10+ for ParamSpec
R = TypeVar("R")
P = ParamSpec("P")


class MindfulLogFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;41m",  # Bold white on red background
    }

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    TIMESTAMP_COLOR = "\033[90m"  # Gray
    LABEL_COLOR = "\033[2m"  # Dim
    VALUE_COLOR = "\033[96m"  # Bright Cyan
    STRING_COLOR = "\033[38;5;183m"  # Light Purple
    NUMBER_COLOR = "\033[38;5;208m"  # Orange
    BOOL_NONE_COLOR = "\033[95m"  # Magenta

    def format(self, record: LogRecord) -> str:
        level_color = self.COLOR_MAP.get(record.levelno, self.RESET)
        timestamp = time.strftime("%H:%M:%S")

        timestamp_str = f"{self.TIMESTAMP_COLOR}{timestamp}{self.RESET}"
        levelname_str = f"{level_color}{record.levelname:<4}{self.RESET}"
        logger_name = f"{self.BOLD}{record.name:<8}{self.RESET}"
        location_str = f"{self.DIM}{record.module}:{record.lineno:<4}{self.RESET}"

        record.message = record.getMessage()
        raw_message = self.formatMessage(record)
        message = self._colorize_message(raw_message, level_color)

        # Combine parts - handle potential multi-line messages from colorize correctly
        first_line, *rest_lines = message.split("\n", 1)
        formatted_message = f"{timestamp_str} {levelname_str} {logger_name} {location_str} - {first_line}"
        if rest_lines:
            padding = " " * (len(timestamp) + 1 + 4 + 1 + 8 + 1 + len(f"{record.module}:{record.lineno:<4}") + 3)
            formatted_message += "\n" + "\n".join(padding + line for line in rest_lines[0].splitlines())
        return formatted_message

    def _colorize_message(self, msg: str, fallback_color: str) -> str:
        """
        Applies beautified formatting to messages of the form 'Label: Value' for any Value type.
        Falls back to level-based color for other messages.
        """
        # Match "Label: Value"
        match = re.match(r"^(.+?):\s(.+)$", msg, re.DOTALL)
        if not match:
            return f"{fallback_color}{msg}{self.RESET}"

        label, value_str = match.groups()
        try:
            # Safely parse Value into a Python literal
            parsed_value = ast.literal_eval(value_str.strip())
            return self._format_value(label, parsed_value)
        except (ValueError, SyntaxError):
            # If parsing fails, treat as raw string
            return self._format_value(label, value_str.strip())

    def _format_value(self, label: str, value: Any) -> str:
        """
        Formats the value based on its type, with consistent indentation and coloring.
        """
        lines = [f"{self.LABEL_COLOR}{label}:{self.RESET}"]

        def format_item(item: Any, indent: int = 2) -> list[str]:
            """Helper to format a single item based on its type."""
            indent_str = " " * indent
            if isinstance(item, dict):
                sub_lines = []
                for k, v in item.items():
                    sub_lines.append(f"{indent_str}{self.BOLD}{k:<6}{self.RESET} → {self._format_single_value(v)}")
                return sub_lines
            elif isinstance(item, (list, tuple)):
                sub_lines = []
                for i, v in enumerate(item):
                    sub_lines.append(f"{indent_str}{i:<2} → {self._format_single_value(v)}")
                return sub_lines
            else:
                return [f"{indent_str}{self._format_single_value(item)}"]

        # Format the top-level value
        if isinstance(value, (dict, list, tuple)):
            lines.extend(format_item(value))
        else:
            lines.append(f"  {self._format_single_value(value)}")

        return "\n".join(lines)

    def _format_single_value(self, value: Any) -> str:
        """
        Formats a single value based on its type.
        """
        if isinstance(value, str):
            return f"{self.STRING_COLOR}{repr(value)}{self.RESET}"
        elif isinstance(value, (int, float)):
            return f"{self.NUMBER_COLOR}{value}{self.RESET}"
        elif isinstance(value, bool) or value is None:
            return f"{self.BOOL_NONE_COLOR}{value}{self.RESET}"
        elif isinstance(value, (list, tuple, dict)):
            # Nested structures are handled by format_item
            return "\n".join(self._format_value("Nested", value).splitlines()[1:])
        else:
            return f"{self.VALUE_COLOR}{value}{self.RESET}"


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
    global _mindful_logger_initialized
    if not _mindful_logger_initialized:
        # Setup handler only once per process, respect debug flag for level
        log_level = logging.DEBUG if debug else logging.INFO
        _mindful_logger.setLevel(log_level)
        if not _mindful_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(MindfulLogFormatter("%(message)s"))
            _mindful_logger.addHandler(handler)
        _mindful_logger_initialized = True
        _mindful_logger.info(f"Mindful logger initialized with level {logging.getLevelName(log_level)}")

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
                _mindful_logger.debug(f"Entering mindful wrapper for {func.__name__}")
            # --- Step 1: Initial setup & get user input ---
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind_partial(*args, **kwargs)
                bound_args.apply_defaults()
                if debug:
                    arg_reprs = {k: repr(v) for k, v in bound_args.arguments.items()}
                    _mindful_logger.debug(f"Arguments bound: {arg_reprs}")
            except TypeError as e:
                _mindful_logger.error(f"Failed to bind arguments for {func.__name__}: {e}")
                raise

            arguments = bound_args.arguments
            if not arguments:
                raise ValueError("Mindful can't wrap a function with no arguments.")
            elif input not in arguments:
                raise ValueError(f"Expected message input '{input}' not found in function signature.")
            original_user_input = bound_args.arguments[input]

            instance_self: Optional[Any] = None  # To hold the instance 'self'
            is_method = "self" in arguments
            if is_method:
                instance_self = arguments["self"]

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
                _mindful_logger.debug(
                    f"Identified user input string: '{original_user_input[:50]}...'"
                )  # Log truncated input

            if debug and tape_deck:
                _mindful_logger.debug(f"Using TapeDeck instance: {tape_deck!r} (id: {id(tape_deck)})")

            # --- Step 2: Retrieve PAST memory tapes ---
            retrieved_memory_messages: List[Dict[str, str]] = []
            mindful_user_input: str
            if debug:
                _mindful_logger.debug(
                    f"Attempting to retrieve relevant memory for query: '{original_user_input[:50]}...'"
                )
            try:
                memory_tapes = tape_deck.retrieve_relevant(original_user_input)
                retrieved_memory_messages = [{"role": t.role, "content": t.content} for t in memory_tapes]
                _mindful_logger.info(f"Retrieved {len(retrieved_memory_messages)} messages from history.")

                if retrieved_memory_messages:
                    memory_log = "\n".join(
                        f"{msg['role'].capitalize()}: {msg['content']}" for msg in retrieved_memory_messages
                    )
                    mindful_user_input = f"<CONTEXT>\n{memory_log}\n</CONTEXT>\nUser: {original_user_input}"
                else:
                    mindful_user_input = f"User: {original_user_input}"

                if debug:
                    _mindful_logger.debug(f"  - Mindful user input: {mindful_user_input}...")
            except Exception as e:
                _mindful_logger.error(f"Failed to retrieve memory tapes: {e}", exc_info=debug)
                mindful_user_input = "User: " + original_user_input + "\n<CONTEXT>No Context Found...</CONTEXT>"

            # --- Step 3: Store User Input Tape ---
            user_tape: Optional[Tape] = None
            if debug:
                _mindful_logger.debug(
                    f"Attempting to store user input tape: role=user, content='{original_user_input[:50]}...'"
                )
            try:
                user_tape = tape_deck.add_tape(content=original_user_input, role="user")
                if debug and user_tape:
                    _mindful_logger.debug(f"Stored user tape successfully: id={user_tape.id}")
            except Exception as e:
                _mindful_logger.error(f"Failed to store user input tape: {e}", exc_info=debug)

            # --- Step 4: Call Original User Function ---
            response: R
            try:
                # Call the user's original function with modified messages
                if debug:
                    _mindful_logger.debug(f"Calling original function: {func.__name__}")
                bound_args.arguments[input] = mindful_user_input
                response = func(*bound_args.args, **bound_args.kwargs)
                if debug:
                    _mindful_logger.debug(f"Original function {func.__name__} execution finished.")
            except Exception as e:
                _mindful_logger.error(
                    f"Error during execution of decorated function {func.__name__}: {e}", exc_info=debug
                )
                raise e

            # --- Step 5: Store Assistant Response & Link ---
            assistant_tape: Optional[Tape] = None
            try:
                assistant_content = cast(str, response)
                if debug:
                    _mindful_logger.debug(
                        f"Attempting to store assistant response tape: role=assistant, content='{assistant_content[:50]}...'"
                    )
                assistant_tape = tape_deck.add_tape(content=assistant_content, role="assistant")
                if debug and assistant_tape:
                    _mindful_logger.debug(f"Stored assistant tape successfully: id={assistant_tape.id}")
            except Exception as e:
                _mindful_logger.error(f"Failed to store assistant response tape: {e}", exc_info=debug)

            # Link user input tape to assistant response tape if both were stored
            if user_tape is not None and assistant_tape is not None:
                if debug:
                    _mindful_logger.debug(
                        f"Attempting to link tapes: user_id={user_tape.id}, assistant_id={assistant_tape.id}"
                    )
                try:
                    if hasattr(user_tape, "id") and hasattr(assistant_tape, "id"):
                        tape_deck.link_tapes(user_tape.id, assistant_tape.id, "response_to")
                        if debug:
                            _mindful_logger.debug("Tapes linked successfully.")
                    else:
                        _mindful_logger.warning("Could not link tapes: ID attribute missing.")
                except Exception as e:
                    _mindful_logger.error(
                        f"Failed to link tapes ({user_tape.id} -> {assistant_tape.id}): {e}", exc_info=debug
                    )
            elif debug:
                _mindful_logger.debug("Skipping tape linking as user or assistant tape is missing.")

            if debug:
                _mindful_logger.debug(f"Exiting mindful wrapper for {func.__name__}.")
            return response  # Return the original function's response

        return wrapper

    return decorator
