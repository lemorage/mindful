from datetime import (
    datetime,
    timezone,
)
import logging
import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from mindful.agent import MindfulAgent
from mindful.memory.tape import Tape
from mindful.models import TapeMetadata
from mindful.vector_store.storage import StorageAdapter

logger = logging.getLogger("mindful")


class TapeDeck:
    """
    High-level interface for managing memory Tapes.

    Orchestrates Agent (for processing) and StorageAdapter (for persistence).
    Hides storage details. Designed to be instantiated by the mindful decorator.
    """

    def __init__(self, vector_store: StorageAdapter, agent: MindfulAgent):
        """
        Initializes the TapeDeck with **injected** dependencies.

        Args:
            storage_adapter: Configured StorageAdapter instance.
            agent: Configured MindfulAgent instance.
        """
        if not isinstance(vector_store, StorageAdapter):
            raise TypeError("storage_adapter must be an instance of StorageAdapter")
        if not isinstance(agent, MindfulAgent):
            raise TypeError("agent must be an instance of MindfulAgent")

        self.storage: StorageAdapter = vector_store
        self.agent: MindfulAgent = agent
        logger.info(f"TapeDeck initialized. Storage: {type(vector_store).__name__}, Agent: {type(agent).__name__}")

        # Configuration for re-ranking retrieval results
        self.rerank_weights = {"w_similarity": 1.0, "w_recency": 0.1, "w_priority": 0.2, "w_access_count": 0.05}
        # Flag to control if access updates are persisted immediately on get
        self.persist_access_on_get: bool = False  # keep reads fast by default

    def add_tape(self, content: str, role: str, source: Optional[str] = "interaction") -> Optional[Tape]:
        """Creates Tape, generates data via Agent, stores via StorageAdapter."""
        logger.debug(f"Adding tape: role='{role}', source='{source}', content='{content[:50]}...'")
        try:
            embedding = self.agent.embed(content)  # agent handles configured model
            if embedding is None:
                logger.error("Embedding failed for new tape content. Aborting add.")
                return None

            # TODO: Let us change it next time!
            category, context, keywords = "", "", []  # type: ignore
            try:
                category, context, keywords = self.agent.generate_metadata(content)  # type: ignore
                if category is None and context is None and keywords == []:
                    raise ValueError("Metadata generation failed: returned (None, None, [])")
                logger.debug("Metadata generated.")
            except Exception as meta_err:
                logger.warning(f"Metadata generation failed: {meta_err}. Continuing with default metadata.")

            tape = Tape(
                content=content,
                role=role,
                source=source,
                embedding_vector=embedding,
                metadata=TapeMetadata(category=category, context=context, keywords=keywords),
            )

            self.storage.add_tape(tape)
            logger.info(f"Added tape {tape.id} via {type(self.storage).__name__}.")
            return tape
        except Exception as e:
            logger.exception(f"Error during add_tape process: {e}", exc_info=True)
            return None

    def get_tape(self, tape_id: str) -> Optional[Tape]:
        """Retrieves a Tape by ID. Updates access count/time IN MEMORY."""
        logger.debug(f"Retrieving tape {tape_id}...")
        try:
            tape = self.storage.get_tape(tape_id)  # delegate fetch
            if tape:
                tape.mark_accessed()

                if self.persist_access_on_get:
                    try:
                        self.storage.update_tape(tape)
                        logger.debug(f"Persisted access update for tape {tape_id}.")
                    except Exception as update_err:
                        logger.error(f"Failed to persist access update for tape {tape_id}: {update_err}")
                        # Continue even if persistence fails, as tape was retrieved
            else:
                logger.debug(f"Tape {tape_id} not found.")
            return tape
        except Exception as e:
            logger.exception(f"Error retrieving tape {tape_id}: {e}", exc_info=True)
            return None

    def update_tape_content(self, tape_id: str, new_content: str, reason: Optional[str] = None) -> Optional[Tape]:
        """Updates content, re-embeds, and persists."""
        logger.debug(f"Updating content for tape {tape_id}...")
        tape = self.storage.get_tape(tape_id)  # fetch existing
        if not tape:
            logger.warning(f"Cannot update tape: Tape {tape_id} not found.")
            return None
        try:
            # Update content in object (manages versions)
            tape.update_content(new_content, reason=reason)
            # Re-embed using agent
            new_embedding = self.agent.embed(new_content)
            if new_embedding is None:
                logger.error(f"Embedding failed for updated content (tape {tape_id}). Update aborted.")
                return None  # don't save update if embedding fails
            tape.embedding_vector = new_embedding
            # TODO: Optional: Regenerate metadata?
            # Persist changes
            self.storage.update_tape(tape)
            logger.info(f"Updated tape {tape_id} via {type(self.storage).__name__}.")
            return tape
        except Exception as e:
            logger.exception(f"Error updating content for tape {tape_id}: {e}", exc_info=True)
            return None

    def delete_tape(self, tape_id: str) -> bool:
        """Deletes a Tape by ID via storage adapter."""
        logger.info(f"Attempting delete for tape {tape_id}.")
        try:
            return self.storage.delete_tape(tape_id)
        except Exception as e:
            logger.exception(f"Error deleting tape {tape_id}: {e}", exc_info=True)
            return False

    def link_tapes(
        self, tape_id1: str, tape_id2: str, relation_type: str = "related", description: Optional[str] = None
    ) -> bool:
        """Creates a bidirectional link between two Tapes."""
        logger.debug(f"Linking {tape_id1} <=> {tape_id2} ({relation_type})...")
        tapes = self.storage.get_tapes_batch([tape_id1, tape_id2])
        tape1 = tapes[0] if tapes and len(tapes) > 0 else None
        tape2 = tapes[1] if tapes and len(tapes) > 1 else None

        if tape1 and tape2:
            try:
                # Define inverse relation
                inverse_relation = f"{relation_type}_by"
                if relation_type == "response_to":
                    inverse_relation = "response"

                # Add links in both directions
                tape1.add_link(tape_id2, f"{relation_type}: {description or ''}".strip())
                tape2.add_link(tape_id1, f"{inverse_relation}: {description or ''}".strip())

                # Persist changes (potential transactionality needed)
                self.storage.update_tape(tape1)
                self.storage.update_tape(tape2)
                logger.info(f"Linked tapes {tape_id1} and {tape_id2} ({relation_type}).")
                return True
            except Exception as e:
                logger.exception(f"Error persisting link between {tape_id1} and {tape_id2}: {e}", exc_info=True)
                return False
        else:
            logger.warning(f"Link failed: Tape {tape_id1 if not tape1 else tape_id2} not found.")
            return False

    def retrieve_relevant(
        self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tape]:
        """Retrieves relevant Tapes using vector search, filtering, and re-ranking."""
        logger.debug(f"Retrieving: k={top_k}, filter={metadata_filter}, query='{query[:50]}...'")
        try:
            # 1. Embed Query
            query_embedding = self.agent.embed(query)
            if query_embedding is None:
                raise ValueError("Query embedding failed.")

            # 2. Initial Search (delegate to each adapter)
            initial_k = max(top_k * 3, 15)
            search_results = self.storage.vector_search(query_embedding, initial_k, metadata_filter)
            if not search_results:
                return []
            logger.debug(f"Initial search got {len(search_results)} candidates.")

            # 3. Fetch Full Tapes (delegate to adapter)
            tape_ids = [tid for tid, score in search_results]
            initial_scores = {tid: score for tid, score in search_results}
            tapes_list = self.storage.get_tapes_batch(tape_ids)
            retrieved_tapes_dict = {t.id: t for t in tapes_list if t is not None}
            if not retrieved_tapes_dict:
                return []

            # 4. Re-ranking
            now = datetime.now(timezone.utc)
            final_scores = {}
            weights = self.rerank_weights
            for tape_id, initial_score in initial_scores.items():
                if tape_id in retrieved_tapes_dict:
                    tape = retrieved_tapes_dict[tape_id]
                    last_accessed_ts = (tape.last_accessed or tape.created_at).replace(
                        tzinfo=timezone.utc
                    )  # handle None
                    days_since = max(0, (now - last_accessed_ts).total_seconds() / 86400.0)
                    recency = math.exp(-0.05 * days_since)
                    priority = (tape.priority - 1) / 9.0
                    access = math.log1p(tape.access_count) / math.log1p(100)

                    final_score = (
                        (weights["w_similarity"] * initial_score)
                        + (weights["w_recency"] * recency)
                        + (weights["w_priority"] * priority)
                        + (weights["w_access_count"] * access)
                    )
                    final_scores[tape_id] = final_score

            # 5. Sort and Return Top K
            sorted_tape_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
            top_tapes = [retrieved_tapes_dict[tid] for tid in sorted_tape_ids[:top_k] if tid in retrieved_tapes_dict]

            logger.info(f"Retrieval complete. Returning {len(top_tapes)} tapes.")
            return top_tapes

        except Exception as e:
            logger.exception(f"Error during retrieve_relevant query='{query[:50]}...': {e}", exc_info=True)
            return []

    def close_storage(self) -> None:
        """Closes the underlying storage connection."""
        logger.info("Attempting to close storage adapter...")
        try:
            self.storage.close()
        except Exception as e:
            logger.error(f"Error closing storage adapter: {e}", exc_info=True)
