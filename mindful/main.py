import ast
import atexit
from functools import wraps
import importlib
import inspect
import logging
import os
import sys
import threading
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

from mindful.agent import MindfulAgent
from mindful.config import MindfulConfig
from mindful.memory.tape import (
    Tape,
    TapeDeck,
)
from mindful.utils import MindfulLogFormatter
from mindful.vector_store.storage import StorageAdapter

logger = logging.getLogger("mindful")

# Flag and lock to ensure default handler setup happens only once per process safely
_mindful_default_handler_configured = False
_mindful_handler_lock = threading.Lock()


# Define TypeVar for the return type and ParamSpec for the parameters
# Requires Python 3.10+ for ParamSpec
R = TypeVar("R")
P = ParamSpec("P")


def mindful(
    input: str, *, debug: bool = False, config: Optional[MindfulConfig] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Factory for the `mindful` decorator. Configures the user input source
    and logging behavior for the 'mindful' package namespace.

    Initializes backend components (storage, agent, tapedeck) implicitly on first use,
    defaulting to ChromaDB storage if not otherwise configured via environment variables.

    Args:
        input (str): The parameter name holding user input.
        debug (bool): Enables DEBUG level logging for the 'mindful' namespace.
        config (Optional[MindfulConfig]): User-provided configuration object.
                                           Overrides env vars and defaults.

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
                handler.setLevel(logging.DEBUG)
                handler.setFormatter(MindfulLogFormatter("%(message)s"))
                package_logger.addHandler(handler)
                package_logger.propagate = False  # prevent propagation to root
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
        deck_init_lock = threading.Lock()

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """The wrapper function that executes around the original."""
            # --- Step 1: Initial setup & get user input ---
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind_partial(*args, **kwargs)
                bound_args.apply_defaults()
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
            tape_deck: TapeDeck
            instance_or_wrapper: Any = None
            is_method = "self" in arguments
            if is_method:
                instance_or_wrapper = arguments["self"]
            else:
                instance_or_wrapper = wrapper

            core_attr_name = "_mindful_core"

            if not hasattr(instance_or_wrapper, core_attr_name):
                with deck_init_lock:  # ensure thread-safe initialization per instance/function
                    if not hasattr(instance_or_wrapper, core_attr_name):
                        wrapper_logger.info(
                            f"Initializing Mindful components (TapeDeck, Storage, Agent) for '{func.__name__}'..."
                        )
                        try:
                            # --- 1. Resolve Configuration (Config Obj -> Env -> Default) ---
                            effective_config = config or MindfulConfig()
                            wrapper_logger.debug(f"Using config object provided: {config is not None}")

                            # Let BaseSettings resolve from env; fallback defaults handled here
                            storage_type = (effective_config.storage_type or "chroma").lower()
                            agent_provider = (effective_config.agent_provider or "openai").lower()
                            vector_size = effective_config.vector_size or 1536

                            wrapper_logger.debug(f"Resolved storage_type: '{storage_type}'")
                            wrapper_logger.debug(f"Resolved agent_provider: '{agent_provider}'")
                            if effective_config.vector_size is None:
                                wrapper_logger.warning(
                                    f"Vector size not specified, defaulting to {vector_size}. Ensure this matches your embedding model!"
                                )

                            # --- Create specific config dicts using resolved values ---
                            storage_config: Dict[str, Any] = {"vector_size": vector_size}
                            if storage_type == "chroma":
                                path = effective_config.chroma_path or f"./mindful_db_{func.__name__}"
                                coll = effective_config.chroma_collection_name or f"tapes_{func.__name__}"
                                storage_config.update({"path": path, "collection_name": coll})
                            elif storage_type == "qdrant":
                                url = effective_config.qdrant_url or "http://localhost:6333"
                                coll = effective_config.qdrant_collection_name or f"tapes_{func.__name__}"
                                api_key = effective_config.qdrant_api_key
                                storage_config.update({"url": url, "collection_name": coll, "api_key": api_key})
                            # Add other storage types...
                            else:
                                raise ValueError(f"Unsupported storage type resolved: {storage_type}")

                            # Agent init kwargs (using non-DI Agent)
                            agent_init_kwargs = effective_config.get_agent_init_kwargs(agent_provider)
                            # Update with env var overrides if necessary (e.g., model names if Agent supports it)

                            # --- 2. Instantiate Components ---
                            wrapper_logger.debug(
                                f"Attempting init: Storage={storage_type}, AgentProvider={agent_provider}"
                            )
                            adapter: StorageAdapter
                            # Use resolved type to import and instantiate
                            if storage_type == "chroma":
                                try:
                                    from mindful.vector_store.chroma import (
                                        ChromaAdapter,
                                    )

                                    adapter = ChromaAdapter()
                                except ImportError:
                                    raise ImportError("`pip install mindful[chroma]` needed.")
                            elif storage_type == "qdrant":
                                try:
                                    # from mindful.vector_store.qdrant import QdrantAdapter
                                    pass
                                    # adapter = QdrantAdapter()
                                except ImportError:
                                    raise ImportError("`pip install mindful[qdrant]` needed.")
                            # Add other elif...
                            else:
                                raise RuntimeError("Invalid resolved storage type")

                            adapter.initialize(storage_config)
                            wrapper_logger.debug(f"Storage adapter '{storage_type}' initialized.")

                            agent = MindfulAgent(**agent_init_kwargs)
                            wrapper_logger.debug(f"MindfulAgent initialized.")

                            # Instantiate the *refactored* TapeDeck, injecting dependencies
                            initialized_deck = TapeDeck(vector_store=adapter, agent=agent)
                            wrapper_logger.debug("TapeDeck initialized.")

                            # --- Store the Initialized TapeDeck ---
                            setattr(instance_or_wrapper, core_attr_name, initialized_deck)
                            wrapper_logger.info(f"Mindful components initialized using {storage_type}.")

                        except Exception as e:
                            wrapper_logger.exception(f"CRITICAL ERROR during Mindful initialization.", exc_info=e)
                            raise RuntimeError(f"Mindful initialization failed: {e}") from e

            # --- Get Initialized TapeDeck ---
            try:
                tape_deck = getattr(instance_or_wrapper, core_attr_name)
                if not isinstance(tape_deck, TapeDeck):
                    raise TypeError("Invalid _mindful_core attribute.")
            except AttributeError:
                raise RuntimeError("Mindful TapeDeck instance not found. Initialization likely failed.")

            if debug:
                # Log the TapeDeck instance being used *after* initialization/retrieval
                wrapper_logger.debug(f"Using initialized TapeDeck: {tape_deck!r} (id: {id(tape_deck)})")

            # --- Step 3: Retrieve PAST memory tapes ---
            retrieved_memory_messages: List[Dict[str, str]] = []
            if debug:
                wrapper_logger.debug(
                    f"Attempting retrieval via TapeDeck for query: '{str(original_user_input)[:100]}...'"
                )
            try:
                relevant_tapes = tape_deck.retrieve_relevant(str(original_user_input))
                retrieved_memory_messages = [{"role": t.role, "content": t.content} for t in relevant_tapes]
            except Exception as e:
                wrapper_logger.error(f"Failed to retrieve memory tapes via TapeDeck: {e}", exc_info=debug)

            # --- Step 4: Store User Input Tape ---
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

            # --- Step 5: Prepare Input & Call Original Function ---
            mindful_user_input: str
            if retrieved_memory_messages:
                memory_log = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in retrieved_memory_messages)
                mindful_user_input = f"<CONTEXT>\n{memory_log}\n</CONTEXT>\nUser: {original_user_input}"
            else:
                mindful_user_input = f"User: {original_user_input}"
            if debug:
                wrapper_logger.debug(
                    f"Prepared mindful_user_input (len {len(mindful_user_input)}): '{mindful_user_input[:150]}...'"
                )

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

            # --- Step 6: Store Assistant Response & Link ---
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
                        link_successful = tape_deck.link_tapes(user_tape_id, assistant_tape_id, "response_to")
                        if link_successful:
                            if debug:
                                wrapper_logger.debug("Tapes linked successfully via TapeDeck.")
                        else:
                            if debug:
                                wrapper_logger.debug("TapeDeck reported link failure (check previous logs).")

                    except Exception as e:
                        wrapper_logger.error(
                            f"Exception during tape linking call ({user_tape_id} -> {assistant_tape_id}): {e}",
                            exc_info=debug,
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
