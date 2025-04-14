import os

from dotenv import (
    find_dotenv,
    load_dotenv,
)


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
