import logging
from pathlib import Path
import threading

logger = logging.getLogger("mindful")

# Thread-safe lock for config directory initialization
_config_dir_lock = threading.Lock()


def get_mindful_config_dir() -> Path:
    """
    Get or create the mindful configuration directory (~/.mindful/).

    This is used for storage backends (e.g., ChromaDB) that store data in the user's home directory.

    Returns:
        Path: The path to the .mindful directory.
    """
    config_dir = Path.home() / ".mindful"
    with _config_dir_lock:
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Mindful config directory ensured at: {config_dir}")
        except Exception as e:
            logger.error(f"Failed to create config directory {config_dir}: {e}")
            raise RuntimeError(f"Cannot create mindful config directory: {e}")
    return config_dir
