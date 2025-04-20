import ast
import logging
from logging import LogRecord
import re
import os
import time
from typing import Any

from dotenv import (
    find_dotenv,
    load_dotenv,
)


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
            return self._format_value(label, value_str.strip())  # Treat as raw string

    def _format_value(self, label: str, value: Any) -> str:
        """
        Formats the value based on its type, with consistent indentation and coloring.
        """
        lines = [f"{self.LABEL_COLOR}{label}:{self.RESET}"]

        def format_item(item: Any, indent: int = 2) -> list[str]:
            """Helper to format a single item based on its type."""
            indent_str = " " * indent
            if isinstance(item, dict):
                sub_lines = [
                    f"{indent_str}{self.BOLD}{k:<6}{self.RESET} → {self._format_single_value(v)}"
                    for k, v in item.items()
                ]
                return sub_lines
            elif isinstance(item, (list, tuple)):
                sub_lines = [f"{indent_str}{i:<2} → {self._format_single_value(v)}" for i, v in enumerate(item)]
                return sub_lines
            else:
                return [f"{indent_str}{self._format_single_value(item)}"]

        if isinstance(value, (dict, list, tuple)):
            lines.extend(format_item(value))
        else:
            lines.append(f"  {self._format_single_value(value)}")
        return "\n".join(lines)

    def _format_single_value(self, value: Any) -> str:
        if isinstance(value, str):
            return f"{self.STRING_COLOR}{repr(value)}{self.RESET}"
        elif isinstance(value, (int, float)):
            return f"{self.NUMBER_COLOR}{value}{self.RESET}"
        elif isinstance(value, bool) or value is None:
            return f"{self.BOOL_NONE_COLOR}{value}{self.RESET}"
        elif isinstance(value, (list, tuple, dict)):
            return "\n".join(self._format_value("Nested", value).splitlines()[1:])
        else:
            return f"{self.VALUE_COLOR}{value}{self.RESET}"


def load_env() -> None:
    """Load environment variables from a .env file if present."""
    _ = load_dotenv(find_dotenv())


def get_api_key(key_name: str) -> str:
    """
    Retrieve an API key from the environment, prioritizing .env files.

    Args:
        key_name (str): The name of the environment variable to fetch (e.g., "OPENAI_API_KEY").

    Returns:
        str: The API key if found.

    Raises:
        ValueError: If the API key is not found.
    """
    load_env()
    api_key = os.getenv(key_name)

    if not api_key:
        raise ValueError(f"API key '{key_name}' not found. Please set it in a .env file or as an environment variable.")

    return api_key
