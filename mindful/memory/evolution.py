from datetime import (
    datetime,
    timedelta,
    timezone,
)
import logging
import random
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from mindful.agent import MindfulAgent
from mindful.memory.tape import Tape
from mindful.vector_store.storage import StorageAdapter

logger = logging.getLogger("mindful")


class MemoryEvolutionManager:
    """
    Manages background tasks for evolving the memory stored via a StorageAdapter.
    """

    def __init__(self, storage_adapter: StorageAdapter, agent: MindfulAgent):
        self.storage = storage_adapter
        self.agent = agent
        logger.info("MemoryEvolutionManager initialized.")

    def run_forgetting_cycle(
        self,
        action: str = "archive",
        max_items_per_run: int = 1000,
        min_idle_period: timedelta = timedelta(days=90),
        max_age: Optional[timedelta] = None,
        min_priority: Optional[int] = 3,
        min_access_count: Optional[int] = 5,
    ) -> None:
        """
        Perform a forgetting cycle to process stale or low-priority tapes.

        This method identifies candidate tapes based on access patterns, age,
        priority, and access count, and performs one of the following actions:
        archive, delete, or decrease their priority. It is designed to
        efficiently manage storage by demoting or removing less important items.

        Args:
            action (str, optional):
                The forgetting action to apply. Must be one of:
                'archive', 'delete', or 'decrease_priority'. Defaults to 'archive'.
            max_items_per_run (int, optional):
                Maximum number of tapes to process during a single run. Defaults to 1000.
            min_idle_period (timedelta, optional):
                Minimum period of inactivity required for a tape to be considered idle.
                Defaults to 90 days.
            max_age (Optional[timedelta], optional):
                Maximum allowed tape age. Tapes older than this will be considered.
                If None, no age limit is applied. Defaults to None.
            min_priority (Optional[int], optional):
                Only tapes with a priority lower than this value will be selected.
                If None, no priority filtering is applied. Defaults to 3.
            min_access_count (Optional[int], optional):
                Only tapes with fewer accesses than this value will be considered.
                If None, no access count filtering is applied. Defaults to 5.

        Notes:
            - Requires the `StorageAdapter` to efficiently support querying by
            multiple criteria (currently uses a slower fallback approach).
            - If no suitable tapes are found, the method exits early.
            - Any errors during tape processing are logged, and processing continues
            with remaining tapes.
        """
        logger.info(f"Starting forgetting cycle: action='{action}', max_items={max_items_per_run}")
        now = datetime.now(timezone.utc)
        idle_before_date = now - min_idle_period if min_idle_period else None
        created_before_date = now - max_age if max_age else None

        try:
            # --- Querying for candidates ---
            # Ideally, the adapter has an efficient method for this complex query.
            # candidate_ids = self.storage.find_tapes_for_evolution(
            #    status="active", # Only consider active tapes
            #    max_last_accessed=idle_before_date,
            #    max_created_at=created_before_date,
            #    max_priority=min_priority - 1 if min_priority is not None else None,
            #    max_access_count=min_access_count - 1 if min_access_count is not None else None,
            #    limit=max_items_per_run
            # )
            # Placeholder: Fetch IDs (inefficient) and filter - requires adapter change
            logger.warning(
                "Using potentially inefficient method to find forgetting candidates. Adapter should implement optimized query."
            )
            all_active_ids = self.storage.find_ids_by_filter(
                {"status": "active"}
            )  # TODO: implement this new adapter method
            if not all_active_ids:
                logger.info("Forgetting cycle: No active tapes found.")
                return

            # Fetch tapes in batches for filtering (still potentially slow)
            candidate_ids = []
            batch_size = 100
            for i in range(0, len(all_active_ids), batch_size):
                batch_ids_to_fetch = all_active_ids[i : i + batch_size]
                tapes_batch = self.storage.get_tapes_batch(batch_ids_to_fetch)
                for tape in tapes_batch:
                    if tape:
                        meets_criteria = True
                        if min_priority is not None and tape.priority >= min_priority:
                            meets_criteria = False
                        if min_access_count is not None and tape.access_count >= min_access_count:
                            meets_criteria = False
                        last_acc = (tape.last_accessed or tape.created_at).replace(tzinfo=timezone.utc)
                        if idle_before_date and last_acc >= idle_before_date:
                            meets_criteria = False
                        created_ts = tape.created_at.replace(tzinfo=timezone.utc)
                        if created_before_date and created_ts >= created_before_date:
                            meets_criteria = False

                        if meets_criteria:
                            candidate_ids.append(tape.id)
                if len(candidate_ids) >= max_items_per_run:
                    break  # stop fetching if we hit the limit

            logger.info(f"Forgetting cycle: Identified {len(candidate_ids)} candidates for action '{action}'.")
            if not candidate_ids:
                return

            # --- Perform Action ---
            actioned_count = 0
            # Fetch full tapes needed for priority decrease action
            tapes_to_action = {}
            if action == "decrease_priority":
                tapes_list = self.storage.get_tapes_batch(candidate_ids)
                tapes_to_action = {t.id: t for t in tapes_list if t}

            for tape_id in candidate_ids:
                try:
                    success = False
                    if action == "archive":
                        # Fetch tape, update status, save update
                        tape = self.storage.get_tape(tape_id)
                        if tape:
                            tape.status = "archived"
                            tape.updated_at = datetime.now(timezone.utc)
                            self.storage.update_tape(tape)
                            logger.debug(f"Archived tape {tape_id}")
                            success = True
                    elif action == "delete":
                        success = self.storage.delete_tape(tape_id)
                        if success:
                            logger.debug(f"Deleted tape {tape_id}")
                    elif action == "decrease_priority":
                        tape = tapes_to_action.get(tape_id)
                        if tape and tape.priority > 1:
                            tape.priority -= 1
                            tape.updated_at = datetime.now(timezone.utc)
                            self.storage.update_tape(tape)
                            logger.debug(f"Decreased priority for tape {tape_id} to {tape.priority}")
                            success = True
                    else:
                        logger.error(f"Unknown forgetting action: {action}")
                        break

                    if success:
                        actioned_count += 1

                except Exception as e:
                    logger.error(f"Error performing action '{action}' on tape {tape_id}: {e}", exc_info=True)

            logger.info(f"Forgetting cycle finished. Performed '{action}' on {actioned_count} tapes.")

        except Exception as e:
            logger.exception(f"Error during forgetting cycle: {e}", exc_info=True)

    def run_summarization_cycle(
        self,
        max_clusters_per_run: int = 10,
        min_cluster_size: int = 5,
        max_cluster_size: int = 20,
        max_source_tape_age: timedelta = timedelta(days=30),
        # Add criteria for selecting seeds (e.g., high priority, recent access?)
    ) -> None:
        """
        Perform a summarization cycle by identifying clusters of related tapes
        and generating summarized tapes.

        This method selects active tapes as seeds, finds semantically similar
        neighbors, summarizes their content using an agent, and creates a
        summary tape. Original tapes are updated to reflect their summarized status.

        Args:
            max_clusters_per_run (int, optional):
                Maximum number of clusters to process in a single run. Defaults to 10.
            min_cluster_size (int, optional):
                Minimum number of tapes required to form a valid cluster. Defaults to 5.
            max_cluster_size (int, optional):
                Maximum number of tapes to include in a single cluster. Defaults to 20.
            max_source_tape_age (timedelta, optional):
                Maximum allowed age for source tapes considered for clustering.
                Tapes older than this threshold may be excluded. Defaults to 30 days.

        Notes:
            - Tapes must have an `embedding_vector` to be considered for clustering.
            - Seed selection currently uses random sampling among active tapes.
            - Summarization relies on an external agent's `summarize_content` method.
            - If summarization or storage fails for a cluster, that cluster is skipped.
            - Each original tape in a successful cluster is linked to the summary
            and marked as summarized.
        """
        logger.info(f"Starting summarization cycle: max_clusters={max_clusters_per_run}")

        try:
            # --- 1. Identify Seed Tapes for Clustering ---
            # Strategy: Find recently accessed, active tapes as potential cluster centers?
            # Or randomly sample active tapes? Needs refinement.
            # Placeholder: Get some recent active tape IDs
            active_ids = self.storage.find_ids_by_filter({"status": "active"})  # TODO: implement adapter method
            if not active_ids:
                return
            seed_candidates = random.sample(active_ids, min(len(active_ids), max_clusters_per_run * 5))
            seed_tapes = self.storage.get_tapes_batch(seed_candidates)
            valid_seed_tapes = [t for t in seed_tapes if t and t.embedding_vector]
            if not valid_seed_tapes:
                return
            logger.debug(f"Identified {len(valid_seed_tapes)} potential seed tapes for summarization.")

            processed_clusters = 0
            now = datetime.now(timezone.utc)
            age_threshold = now - max_source_tape_age

            for seed_tape in valid_seed_tapes:
                if processed_clusters >= max_clusters_per_run:
                    break

                # --- 2. Find Neighbors for Seed Tape (Potential Cluster) ---
                # Find tapes semantically similar to the seed, excluding very old ones maybe
                logger.debug(f"Finding neighbors for seed tape {seed_tape.id}...")
                # Filter could exclude already summarized tapes and very old tapes
                # filter_dict = {"status": "active", "created_after": age_threshold}
                filter_dict = {"status": "active"}
                neighbors_found = self.storage.vector_search(
                    query_vector=seed_tape.embedding_vector, top_k=max_cluster_size, filter_dict=filter_dict  # type: ignore
                )
                # Get IDs, excluding the seed itself, filter by similarity threshold?
                similarity_threshold = 0.75
                cluster_ids = [
                    nid for nid, score in neighbors_found if nid != seed_tape.id and score >= similarity_threshold
                ]
                # Add seed tape ID back
                cluster_ids.insert(0, seed_tape.id)

                if len(cluster_ids) < min_cluster_size:
                    logger.debug(
                        f"Skipping seed {seed_tape.id}: Not enough similar neighbors ({len(cluster_ids)} found)."
                    )
                    continue

                logger.info(f"Found potential cluster of size {len(cluster_ids)} around seed {seed_tape.id}.")

                # --- 3. Retrieve Content & Summarize ---
                cluster_tapes = self.storage.get_tapes_batch(cluster_ids)
                valid_cluster_tapes = [t for t in cluster_tapes if t]
                if len(valid_cluster_tapes) < min_cluster_size:
                    continue  # Check again after fetch

                contents_to_summarize = [f"{t.role}: {t.content}" for t in valid_cluster_tapes]
                # Format for LLM summarization prompt
                combined_content = "\n---\n".join(contents_to_summarize)
                # Use Agent to summarize
                summary_text = self.agent.summarize_content(combined_content)  # TODO: needs implementation

                if not summary_text:
                    logger.warning(f"Summarization failed for cluster around {seed_tape.id}.")
                    continue

                # --- 4. Create and Store Summary Tape ---
                # Store summary tape before updating originals
                summary_tape = self.add_tape(
                    content=summary_text,
                    role="agent_summary",  # special role
                    source=f"summary_of_{len(valid_cluster_tapes)}_tapes_around_{seed_tape.id[:8]}",
                )
                if not summary_tape:
                    logger.error(f"Failed to store summary tape for cluster around {seed_tape.id}.")
                    continue

                # --- 5. Link Summary & Update Originals ---
                processed_clusters += 1
                updated_originals = 0
                for original_tape in valid_cluster_tapes:
                    try:
                        # Link original to summary
                        self.link_tapes(original_tape.id, summary_tape.id, "summarized_by")
                        # Link summary to original
                        self.link_tapes(summary_tape.id, original_tape.id, "summary_of")

                        # Update original tape status (optional)
                        original_tape.status = "summarized"  # or "archived"
                        original_tape.updated_at = datetime.now(timezone.utc)
                        self.storage.update_tape(original_tape)
                        updated_originals += 1
                    except Exception as e:
                        logger.error(
                            f"Error updating/linking original tape {original_tape.id} after summarization: {e}"
                        )

                logger.info(
                    f"Successfully created summary tape {summary_tape.id} and updated/linked {updated_originals} source tapes."
                )

            logger.info(f"Summarization cycle finished. Processed {processed_clusters} clusters.")

        except Exception as e:
            logger.exception(f"Error during summarization cycle: {e}", exc_info=True)

    # --- TODO: We might need corresponding TapeDeck methods ---
    # These might call storage adapter methods directly or use TapeDeck's existing ones
    def add_tape(self, content: str, role: str, source: Optional[str]) -> Optional[Tape]:
        # Need access to TapeDeck's add_tape or replicate its logic using self.agent and self.storage
        # For simplicity, assume MemoryEvolutionManager is GIVEN a TapeDeck instance instead of adapter/agent
        # Or pass TapeDeck instance to the cycle methods?
        logger.warning("add_tape within Evolution Manager needs proper implementation using TapeDeck/Storage/Agent.")
        return None

    def link_tapes(self, id1: str, id2: str, relation: str) -> None:
        logger.warning("link_tapes within Evolution Manager needs proper implementation using TapeDeck/Storage.")
        pass
