from datetime import datetime
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
import uuid

from mindful.memory.tape import Tape
from mindful.vector_store.storage import StorageAdapter

logger = logging.getLogger("mindful")

try:
    from qdrant_client import (
        QdrantClient,
        models,
    )
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        OrderBy,
        PointStruct,
        Range,
        VectorParams,
    )

    qdrant_client_installed = True
except ImportError:
    QdrantClient = None
    models = None
    qdrant_client_installed = False
    logger.warning("Qdrant client not installed (`pip install qdrant-client`). QdrantAdapter unavailable.")


# Helper function to convert Tape to Qdrant payload and back
def _tape_to_qdrant_payload(tape: Tape) -> Dict[str, Any]:
    """Converts Tape model fields to Qdrant payload, handling datetimes."""
    payload = tape.model_dump(exclude={"id", "embedding_vector"})
    for key, value in payload.items():
        if isinstance(value, datetime):
            payload[key] = value.isoformat()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], datetime):
            payload[key] = [dt.isoformat() for dt in value]
        # Qdrant typically handles standard JSON types (str, int, float, bool, list, dict)
    return payload


def _qdrant_hit_to_tape(hit: models.ScoredPoint | models.Record) -> Optional[Tape]:  # type: ignore[no-any-unimported]
    """Converts a Qdrant search hit or record back to a Tape object."""
    payload = hit.payload or {}
    # Convert ISO strings back to datetime
    for key, value in payload.items():
        if isinstance(value, str):
            try:
                # Attempt to parse as ISO datetime
                # This is basic, might need adjustment based on actual fields
                if key in ["created_at", "updated_at", "last_accessed", "expires_at"]:
                    payload[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
    try:
        # Qdrant point ID can be UUID string or integer, ensure consistency
        # Assuming Tape.id is always stored/retrieved as string representation of UUID
        tape_id = str(hit.id)
        # Handle both ScoredPoint (has vector) and Record (might not)
        vector = getattr(hit, "vector", None) or []

        # Reconstruct Tape - make sure all fields expected by Tape.__init__ are present
        # Use payload.get('field', default_value) if fields might be missing
        tape_data = {
            "id": tape_id,
            "embedding_vector": vector,
            **payload,
        }
        # Add default values for any Optional fields not in payload if needed
        # e.g., tape_data.setdefault("category", None)
        return Tape.model_validate(tape_data)

    except Exception as e:
        logger.error(f"Failed to convert Qdrant hit (id={hit.id}) to Tape: {e}", exc_info=True)
        return None


def _build_qdrant_filter(filter_dict: Dict[str, Any]) -> Optional[Filter]:  # type: ignore[no-any-unimported]
    """Translates a generic filter dict to Qdrant's Filter model."""
    must_conditions = []
    for key, value in filter_dict.items():
        if isinstance(value, dict) and len(value) == 1:  # handle explicit operators
            op, val = next(iter(value.items()))
            if op == "$eq":
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            elif op == "$in" and isinstance(val, list):
                must_conditions.append(FieldCondition(key=key, match=MatchAny(any=val)))
            elif op == "$gte":
                val = val.isoformat() if isinstance(val, datetime) else val
                must_conditions.append(FieldCondition(key=key, range=Range(gte=val)))
            elif op == "$lte":
                val = val.isoformat() if isinstance(val, datetime) else val
                must_conditions.append(FieldCondition(key=key, range=Range(lte=val)))
            else:
                logger.warning(f"Unsupported operator '{op}' for key '{key}' in Qdrant filter")
        elif key.endswith("_gte"):
            field = key[:-4]
            val = value.isoformat() if isinstance(field, datetime) else value
            must_conditions.append(FieldCondition(key=field, range=Range(gte=val)))
        elif key.endswith("_lte"):
            field = key[:-4]
            val = value.isoformat() if isinstance(field, datetime) else value
            must_conditions.append(FieldCondition(key=field, range=Range(lte=val)))
        elif key == "keywords_contain":
            if isinstance(value, list):
                must_conditions.append(FieldCondition(key="keywords", match=MatchAny(any=value)))
            else:
                must_conditions.append(FieldCondition(key="keywords", match=MatchValue(value=value)))
        else:
            # Default to equality match
            must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=must_conditions) if must_conditions else None


class QdrantAdapter(StorageAdapter):
    """Storage adapter implementation for Qdrant vector database."""

    def __init__(self) -> None:
        if not qdrant_client_installed:
            raise ImportError("Qdrant client not installed. Please run `pip install qdrant-client`")
        self.client: Optional[QdrantClient] = None  # type: ignore[no-any-unimported]
        self.collection_name: Optional[str] = None
        self.vector_size: Optional[int] = None
        self.distance_metric: models.Distance = Distance.COSINE  # type: ignore[no-any-unimported]

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initializes Qdrant client and ensures collection exists.

        Expected config keys:
        - url (str): Qdrant URL (e.g., "http://localhost:6333" or Qdrant Cloud URL). Required.
        - port (int): Optional, often included in url.
        - api_key (Optional[str]): API key for Qdrant Cloud or secured instances.
        - collection_name (str): Name of the collection. Required.
        - vector_size (int): Dimension of embedding vectors. Required.
        - distance (str): Optional distance metric ('Cosine', 'Euclid', 'Dot'). Defaults to Cosine.
        - create_collection_if_not_exists (bool): Optional, defaults True.
        - HNSW/other indexing params: Optional, for tuning collection creation.
        """
        logger.info(f"Initializing QdrantAdapter with config: {config}")
        try:
            self.collection_name = config["collection_name"]
            self.vector_size = config["vector_size"]
            distance_str = config.get("distance", "Cosine").upper()
            if distance_str == "COSINE":
                self.distance_metric = Distance.COSINE
            elif distance_str == "EUCLID":
                self.distance_metric = Distance.EUCLID
            elif distance_str == "DOT":
                self.distance_metric = Distance.DOT
            else:
                raise ValueError(f"Unsupported distance metric: {distance_str}")

            self.client = QdrantClient(
                url=config["url"],
                port=config.get("port", 6333 if "localhost" in config["url"] else 443),
                api_key=config.get("api_key"),
                # Add timeout configs etc.
            )
            logger.debug("Qdrant client created.")

            # Check and potentially create collection
            create = config.get("create_collection_if_not_exists", True)
            collection_exists = False
            try:
                self.client.get_collection(collection_name=self.collection_name)
                collection_exists = True
                logger.info(f"Connected to existing Qdrant collection '{self.collection_name}'")
            except Exception as e:
                # More specific exception catching is better (e.g., UnexpectedResponse with 404)
                logger.info(f"Qdrant collection '{self.collection_name}' not found ({e}).")

            if not collection_exists and create:
                logger.info(f"Attempting to create Qdrant collection '{self.collection_name}'...")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance_metric),
                    # TODO: Add payload index schema creation based on Tape fields for filtering
                    # self.client.create_payload_index(...) for category, role, priority etc.
                )
                # Create payload indexes for efficient filtering
                index_fields = [
                    ("status", models.PayloadFieldType.KEYWORD),
                    ("priority", models.PayloadFieldType.INTEGER),
                    ("role", models.PayloadFieldType.KEYWORD),
                    ("category", models.PayloadFieldType.KEYWORD),
                    ("created_at", models.PayloadFieldType.DATETIME),
                    ("last_accessed", models.PayloadFieldType.DATETIME),
                ]
                for field_name, field_type in index_fields:
                    try:
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                        )
                        logger.debug(
                            f"Created payload index for field '{field_name}' in collection '{self.collection_name}'"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create payload index for field '{field_name}': {e}")
                logger.info(f"Created Qdrant collection '{self.collection_name}' with payload indexes.")
            elif not collection_exists and not create:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' does not exist and creation is disabled."
                )
            elif not collection_exists and not create:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' does not exist and creation is disabled."
                )

        except KeyError as e:
            logger.error(f"Missing required key in Qdrant config: {e}")
            raise ValueError(f"Missing required key in Qdrant config: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize QdrantAdapter: {e}", exc_info=True)
            self.client = None
            raise ConnectionError(f"Failed to initialize QdrantAdapter: {e}") from e

    def add_tape(self, tape: Tape) -> None:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")
        if not tape.embedding_vector:
            raise ValueError("Tape must have an embedding vector to be added to Qdrant.")
        # Use Tape.id (UUID string) as Qdrant point ID
        point = PointStruct(id=tape.id, vector=tape.embedding_vector, payload=_tape_to_qdrant_payload(tape))
        try:
            # Use upsert for simplicity (add or update)
            self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)
            logger.debug(f"Upserted tape {tape.id} to Qdrant.")
        except Exception as e:
            logger.error(f"Failed to upsert tape {tape.id} to Qdrant: {e}", exc_info=True)
            raise

    def add_tapes_batch(self, tapes: List[Tape]) -> None:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")
        points_batch = []
        skipped_count = 0
        for tape in tapes:
            if tape.embedding_vector:
                points_batch.append(
                    PointStruct(id=tape.id, vector=tape.embedding_vector, payload=_tape_to_qdrant_payload(tape))
                )
            else:
                logger.warning(f"Skipping tape {tape.id} in batch add: Missing embedding vector.")
                skipped_count += 1
        if points_batch:
            logger.info(f"Adding batch of {len(points_batch)} tapes to Qdrant (skipped {skipped_count})...")
            try:
                # Adjust batch size if needed for Qdrant limits/performance
                self.client.upsert(collection_name=self.collection_name, points=points_batch, wait=True)
                logger.debug(f"Upserted batch of {len(points_batch)} tapes to Qdrant.")
            except Exception as e:
                logger.error(f"Failed to upsert tape batch to Qdrant: {e}", exc_info=True)
                raise
        elif skipped_count > 0:
            logger.warning("No tapes in batch had embeddings, nothing added.")

    def get_tape(self, tape_id: str) -> Optional[Tape]:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[tape_id],
                with_payload=True,
                with_vectors=True,  # need vector if tape object requires it
            )
            if results:
                hit = results[0]
                return _qdrant_hit_to_tape(hit)
            else:
                logger.debug(f"Tape {tape_id} not found in Qdrant.")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve tape {tape_id} from Qdrant: {e}", exc_info=True)
            return None

    def get_tapes_batch(self, tape_ids: List[str]) -> List[Optional[Tape]]:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")
        if not tape_ids:
            return []
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name, ids=tape_ids, with_payload=True, with_vectors=True
            )
            # Qdrant might return fewer results than IDs requested if some don't exist
            # Create a map for quick lookup and preserve order/None for missing
            results_map = {str(hit.id): hit for hit in results}
            final_tapes = []
            for tid in tape_ids:
                hit = results_map.get(tid)
                if hit:
                    final_tapes.append(_qdrant_hit_to_tape(hit))
                else:
                    final_tapes.append(None)
                    logger.debug(f"Tape {tid} not found in Qdrant during batch get.")
            return final_tapes
        except Exception as e:
            logger.error(f"Failed to retrieve tape batch from Qdrant: {e}", exc_info=True)
            # Return Nones based on input length on error? Or raise?
            return [None] * len(tape_ids)

    def update_tape(self, tape: Tape) -> None:
        # Qdrant's upsert handles updates naturally
        logger.debug(f"Updating tape {tape.id} in Qdrant using upsert.")
        self.add_tape(tape)

    def delete_tape(self, tape_id: str) -> bool:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")
        try:
            # Use PointIdsList for selector
            response = self.client.delete(
                collection_name=self.collection_name, points_selector=models.PointIdsList(points=[tape_id]), wait=True
            )
            # Check status (COMPLETED means successful deletion or point didn't exist)
            if response.status == models.UpdateStatus.COMPLETED:
                logger.debug(f"Delete operation completed for tape {tape_id} in Qdrant.")
                # Qdrant delete doesn't easily tell if it *actually* deleted something
                # vs. the ID not existing. Return True if operation completed.
                return True
            else:
                logger.warning(f"Delete operation for tape {tape_id} in Qdrant returned status: {response.status}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete tape {tape_id} from Qdrant: {e}", exc_info=True)
            return False

    def vector_search(
        self, query_vector: List[float], top_k: int, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")

        qdrant_filter = _build_qdrant_filter(filter_dict) if filter_dict else None
        logger.debug(f"Performing Qdrant vector search: k={top_k}, filter={qdrant_filter}")

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_vectors=False,  # don't need vectors in results
                with_payload=False,  # don't need payload either
            )
            # Convert result IDs to string consistently
            return [(str(hit.id), hit.score) for hit in search_result]
        except Exception as e:
            logger.error(f"Vector search failed in Qdrant: {e}", exc_info=True)
            return []

    def get_all_tape_ids(self) -> List[str]:
        if not self.client or not self.collection_name:
            raise ConnectionError("Qdrant client not initialized")
        try:
            all_ids: List[str] = []
            scroll_offset = None
            batch_size = 1000
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=scroll_offset,
                    with_payload=False,
                    with_vectors=False,
                )
                all_ids.extend(str(point.id) for point in points)
                scroll_offset = next_offset
                if not next_offset or not points:
                    break
            logger.debug(f"Fetched {len(all_ids)} tape IDs from Qdrant")
            return all_ids
        except Exception as e:
            logger.error(f"Failed to get all tape IDs from Qdrant: {e}", exc_info=True)
            return []

    def find_ids_by_filter(
        self,
        filter_dict: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = True,
    ) -> List[str]:
        """
        Finds Tape IDs based solely on metadata filtering criteria in Qdrant.

        Args:
            filter_dict: Dictionary defining metadata filters (e.g., {'status': 'active', 'priority_lte': 3}).
                        Supports operators like '$eq', '$in', '$gte', '$lte', and suffix-based '_gte', '_lte'.
            limit: Maximum number of IDs to return. If None, returns all matching IDs.
            offset: Number of IDs to skip for pagination. Defaults to 0 if None.
            sort_by: Payload field to sort results by (e.g., 'created_at'). Uses Qdrant native sorting if supported.
            sort_desc: If True, sort in descending order; otherwise, ascending.

        Returns:
            List[str]: A list of Tape IDs matching the filter criteria, potentially limited, offset, and sorted.
                    Returns an empty list if none found or on error (errors are logged).
        """
        if not self.client or not self.collection_name:
            logger.error("Qdrant client not initialized")
            return []

        # Validate inputs
        offset = offset or 0
        if offset < 0 or (limit is not None and limit < 0):
            logger.error(f"Invalid pagination parameters: offset={offset}, limit={limit}")
            return []
        supported_sort_fields = {"created_at", "last_accessed", "priority", "status"}
        if sort_by and sort_by not in supported_sort_fields:
            logger.warning(f"Unsupported sort_by field '{sort_by}' in Qdrant. Falling back to unsorted results.")
            sort_by = None

        try:
            # Build Qdrant filter
            qdrant_filter = _build_qdrant_filter(filter_dict) if filter_dict else None
            logger.debug(
                f"Qdrant find_ids_by_filter: filter={qdrant_filter}, limit={limit}, offset={offset}, sort_by={sort_by}, sort_desc={sort_desc}"
            )

            # Prepare sorting
            order_by = None
            if sort_by:
                direction = "desc" if sort_desc else "asc"
                order_by = OrderBy(key=sort_by, direction=direction)

            # Scroll through results
            all_ids: List[str] = []
            scroll_offset = None
            batch_size = min(1000, limit) if limit else 1000
            remaining_limit = limit

            while True:
                current_limit = min(batch_size, remaining_limit) if remaining_limit else batch_size
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    query_filter=qdrant_filter,
                    limit=current_limit,
                    offset=scroll_offset if scroll_offset else offset,
                    with_payload=False,
                    with_vectors=False,
                    order_by=order_by,
                )
                all_ids.extend(str(point.id) for point in points)
                scroll_offset = next_offset
                if not points or not next_offset or (remaining_limit and len(all_ids) >= remaining_limit):
                    break
                if remaining_limit:
                    remaining_limit -= len(points)

            if limit is not None and len(all_ids) > limit:
                all_ids = all_ids[:limit]

            logger.debug(f"Returning {len(all_ids)} IDs after applying filter and pagination")
            return all_ids

        except Exception as e:
            logger.error(f"Failed to find IDs by filter in Qdrant: {e}", exc_info=True)
            return []

    def close(self) -> None:
        if self.client:
            logger.info("Closing Qdrant client.")
            # Qdrant client might manage connections automatically,
            # but explicit close is available for http client if needed.
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Exception during Qdrant client close: {e}")
        self.client = None
