import logging
from pathlib import Path
import threading
import typer
import importlib.metadata

app = typer.Typer(add_completion=False)


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


def version_callback(value: bool) -> None:
    if value:
        try:
            ver = importlib.metadata.version("mindful")
        except importlib.metadata.PackageNotFoundError:
            ver = "unknown (not installed via package)"
        typer.secho(f"Mindful version: {ver}", fg=typer.colors.BRIGHT_GREEN, bold=True)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the current version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Mindful: See through your long-term, self-evolving AI agent memories 🧠
    """
    pass


@app.command()
def info() -> None:
    """Show package info."""
    typer.echo("Mindful: See through your long-term, self-evolving AI agent memories 🧠")
