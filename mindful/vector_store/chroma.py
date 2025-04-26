from datetime import datetime
import json
import logging
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)
import uuid

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from mindful.memory.tape import Tape
from mindful.vector_store.storage import StorageAdapter

logger = logging.getLogger("mindful")

try:
    import chromadb
    from chromadb.api.types import (
        EmbeddingFunction,  # For potential custom embedding func
    )
    from chromadb.utils import (
        embedding_functions,  # For default embedding functions
    )

    chromadb_installed = True
except ImportError:
    chromadb = None
    EmbeddingFunction = None
    embedding_functions = None
    chromadb_installed = False
    logger.warning("ChromaDB client not installed (`pip install chromadb`). ChromaAdapter unavailable.")


# Chroma stores metadata separately from the 'document' (which is often the content)
# We need functions to pack/unpack Tape fields into Chroma's metadata dict and document.
def _tape_to_chroma_meta_and_doc(tape: Tape) -> Tuple[str, Dict[str, Any]]:
    """Separates Tape content (document) from other fields (metadata)."""
    metadata = tape.model_dump(exclude={"id", "embedding_vector", "content"})
    # Convert non-primitive types in metadata to Chroma-compatible types (str, int, float, bool)
    for key, value in metadata.items():
        if isinstance(value, datetime):
            metadata[key] = value.isoformat()  # Store datetime as ISO string
        elif isinstance(value, (list, dict)):
            # Chroma metadata supports basic types. Store lists/dicts as JSON strings?
            # Or flatten if possible. Storing as JSON string is simpler.
            try:
                metadata[key] = json.dumps(value)
            except TypeError:
                logger.warning(f"Could not JSON serialize metadata field '{key}' for Chroma. Storing as string.")
                metadata[key] = str(value)
        elif not isinstance(value, (str, int, float, bool)) and value is not None:
            # Handle other potential types
            metadata[key] = str(value)

    # Keep only compatible types, remove None values as Chroma might dislike them
    compatible_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

    return tape.content, compatible_metadata


def _chroma_result_to_tape(
    ids: List[str],
    metadatas: List[Optional[Dict[str, Any]]],
    documents: List[Optional[str]],
    distances: Optional[List[float]] = None,  # Only from query
    embeddings: Optional[List[Optional[List[float]]]] = None,  # Only from get with include=['embeddings']
) -> List[Optional[Tape]]:
    """Converts Chroma query/get results back to Tape objects."""
    tapes: List[Optional[Tape]] = []
    num_results = len(ids)

    for i in range(num_results):
        tape_id = ids[i]
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        doc = documents[i] if documents and i < len(documents) else None
        embedding = embeddings[i] if embeddings and i < len(embeddings) else None
        logger.debug(
            f"Processing embedding for tape {tape_id}, type: {type(embedding).__name__}, sample: {embedding[:5] if isinstance(embedding, list) else embedding}"
        )

        if doc is None or meta is None:
            logger.warning(f"Missing document or metadata for Chroma ID {tape_id}. Cannot reconstruct Tape.")
            tapes.append(None)
            continue

        try:
            # Reconstruct tape data from metadata and document
            tape_data: Dict[str, Any] = {}
            # Deserialize complex fields stored as JSON strings
            for key, value in meta.items():
                if key in ["created_at", "updated_at", "last_accessed", "expires_at"]:
                    # Handle datetime fields
                    if isinstance(value, str):
                        try:
                            tape_data[key] = datetime.fromisoformat(value)
                        except ValueError as e:
                            logger.warning(f"Invalid datetime format for {key} in tape {tape_id}: {value}")
                            tape_data[key] = None  # or set a default datetime
                    else:
                        logger.warning(f"Expected string for {key} in tape {tape_id}, got {type(value).__name__}")
                        tape_data[key] = None  # or handle differently
                elif key in ["keywords", "links", "related_queries", "versions", "metadata"]:
                    # Deserialize JSON strings
                    if isinstance(value, str):
                        try:
                            tape_data[key] = json.loads(value)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON for {key} in tape {tape_id}: {value}")
                            tape_data[key] = value  # keep as string if parsing fails
                    else:
                        tape_data[key] = value  # keep as is if not a string
                else:
                    tape_data[key] = value  # regular field, no parsing needed

            # Add core fields
            tape_data["id"] = tape_id
            tape_data["content"] = doc
            tape_data["embedding_vector"] = embedding

            # Ensure required fields have defaults if missing
            for field in ["created_at", "updated_at", "last_accessed", "expires_at"]:
                if field not in tape_data or tape_data[field] is None:
                    tape_data[field] = None  # TODO: set a default datetime, e.g., datetime.now()
            # Example: tape_data.setdefault("role", "unknown") for other fields

            logger.debug(
                f"tape_data for {tape_id} before validation: id={tape_data['id']}, metadata_type={type(tape_data.get('metadata'))}, metadata_sample={tape_data.get('metadata')}, embedding_vector_type={type(tape_data['embedding_vector'])}"
            )
            tapes.append(Tape.model_validate(tape_data))

        except Exception as e:
            logger.error(f"Failed to convert Chroma result (id={tape_id}) to Tape: {e}", exc_info=True)
            tapes.append(None)  # append None if conversion fails

    return tapes


def _build_chroma_where_filter(filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translates a generic filter dict to Chroma's 'where' clause format."""
    # Chroma 'where' uses operators like $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    # See Chroma docs for full syntax.
    where_filter = {}
    for key, value in filter_dict.items():
        if key == "role":
            where_filter[key] = {"$eq": value}
        elif key == "category":
            where_filter[key] = {"$eq": value}
        elif key == "priority_gte":
            where_filter["priority"] = {"$gte": value}
        elif key == "priority_lte":
            where_filter["priority"] = {"$lte": value}
        elif key == "keywords_contain":  # Check if keywords list (stored as JSON string?) contains a value
            # This is hard if stored as JSON string. If keywords were stored as a flat list
            # directly in metadata (if Chroma supports it), we might use '$contains' or '$in'.
            # Assuming JSON string, filtering is limited without parsing.
            logger.warning("Complex keyword filtering not implemented for Chroma JSON storage.")
            # Example: Simple exact match on the serialized list (unlikely useful)
            # try: where_filter["keywords"] = {"$eq": json.dumps(value)} except: pass
        elif isinstance(value, dict) and len(value) == 1:  # Handle pre-formatted chroma operators
            op = list(value.keys())[0]
            if op in ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]:
                where_filter[key] = value
            else:
                logger.warning(f"Unsupported operator '{op}' in Chroma filter dict for key '{key}'")
        elif key == "id":
            where_filter["id"] = {"$eq": value}  # Chroma filters on metadata, ID isn't metadata by default
            logger.warning("Filtering by 'id' might require 'id' to be stored in metadata in Chroma.")
        else:
            # Default to equality check if no operator specified
            where_filter[key] = {"$eq": value}

    return where_filter if where_filter else None


class ChromaAdapter(StorageAdapter):
    """Storage adapter implementation for ChromaDB."""

    def __init__(self) -> None:
        if not chromadb_installed:
            raise ImportError("ChromaDB client not installed. Please run `pip install chromadb`")
        self.client: Optional["ClientAPI"] = None  # type: ignore[no-any-unimported] # persistent client or ephemeral?
        self.collection: Optional["Collection"] = None  # type: ignore[no-any-unimported]
        # Chroma often uses its own embedding function or requires SentenceTransformer embeddings
        # We rely on embeddings generated by the MindfulAgent externally for consistency.
        # self.embedding_function: Optional[EmbeddingFunction] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initializes ChromaDB client and gets/creates the collection.

        Expected config keys:
        - path (str): Path for persistent storage (e.g., "./mindful_chroma_db"). Required.
        - collection_name (str): Name of the collection. Required.
        - tenant (str): Optional tenant name (defaults to chromadb.DEFAULT_TENANT).
        - database (str): Optional database name (defaults to chromadb.DEFAULT_DATABASE).
        - create_collection_if_not_exists (bool): Optional, defaults True.
        - embedding_model_name (Optional[str]): Name of model used for embeddings (for metadata).
          Chroma needs this to know how to compare vectors, but we provide vectors directly.
          It's good practice to store it in collection metadata if possible.
        """
        logger.info(f"Initializing ChromaAdapter with config: {config}")
        try:
            path = config["path"]
            collection_name = config["collection_name"]
            tenant = config.get("tenant", chromadb.DEFAULT_TENANT)
            database = config.get("database", chromadb.DEFAULT_DATABASE)
            create = config.get("create_collection_if_not_exists", True)
            # embedding_model_name = config.get("embedding_model_name")

            # Use PersistentClient for local storage
            self.client = chromadb.PersistentClient(path=path, tenant=tenant, database=database)
            logger.debug(f"Chroma PersistentClient initialized for path: {path}")

            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Attached to existing Chroma collection '{collection_name}'")
                # TODO: Validate existing collection metadata (e.g., embedding func name) if needed
            except Exception as e:  # Catch specific exception for collection not found
                logger.info(f"Chroma collection '{collection_name}' not found ({e}).")
                if create:
                    logger.info(f"Attempting to create Chroma collection '{collection_name}'...")
                    # We provide embeddings, so Chroma doesn't need its own embedding function usually.
                    # Can provide metadata like {"hnsw:space": "cosine"} for index tuning.
                    # Store embedding model name for reference if desired.
                    collection_metadata = {"hnsw:space": "cosine"}  # default to cosine distance
                    # if embedding_model_name: collection_metadata["embedding_model"] = embedding_model_name

                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata=collection_metadata,
                        # embedding_function=None # explicitly state we provide embeddings
                    )
                    logger.info(f"Created Chroma collection '{collection_name}'.")
                else:
                    raise RuntimeError(
                        f"Chroma collection '{collection_name}' does not exist and creation is disabled."
                    )

        except KeyError as e:
            logger.error(f"Missing required key in Chroma config: {e}")
            raise ValueError(f"Missing required key in Chroma config: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize ChromaAdapter: {e}", exc_info=True)
            self.client = None
            self.collection = None
            raise ConnectionError(f"Failed to initialize ChromaAdapter: {e}") from e

    def add_tape(self, tape: Tape) -> None:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")
        if not tape.embedding_vector:
            raise ValueError("Tape must have an embedding vector to be added to Chroma.")

        doc, meta = _tape_to_chroma_meta_and_doc(tape)
        try:
            self.collection.upsert(
                ids=[tape.id],
                embeddings=[tape.embedding_vector],
                documents=[doc],
                metadatas=[meta],
            )
            logger.debug(f"Upserted tape {tape.id} to Chroma.")
        except Exception as e:
            logger.error(f"Failed to upsert tape {tape.id} to Chroma: {e}", exc_info=True)
            raise

    def add_tapes_batch(self, tapes: List[Tape]) -> None:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")
        ids_batch, embeddings_batch, docs_batch, metas_batch = [], [], [], []
        skipped_count = 0
        for tape in tapes:
            if tape.embedding_vector:
                doc, meta = _tape_to_chroma_meta_and_doc(tape)
                ids_batch.append(tape.id)
                embeddings_batch.append(tape.embedding_vector)
                docs_batch.append(doc)
                metas_batch.append(meta)
            else:
                logger.warning(f"Skipping tape {tape.id} in batch add: Missing embedding vector.")
                skipped_count += 1

        if ids_batch:
            logger.info(f"Adding batch of {len(ids_batch)} tapes to Chroma (skipped {skipped_count})...")
            try:
                # Upsert the batch
                self.collection.upsert(
                    ids=ids_batch, embeddings=embeddings_batch, documents=docs_batch, metadatas=metas_batch
                )
                logger.debug(f"Upserted batch of {len(ids_batch)} tapes to Chroma.")
            except Exception as e:
                logger.error(f"Failed to upsert tape batch to Chroma: {e}", exc_info=True)
                raise  # Re-raise
        elif skipped_count > 0:
            logger.warning("No tapes in batch had embeddings, nothing added.")

    def get_tape(self, tape_id: str) -> Optional[Tape]:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")
        try:
            results = self.collection.get(
                ids=[tape_id],
                include=["metadatas", "documents", "embeddings"],
            )
            # Check if ID was actually returned
            if results and results["ids"] and results["ids"][0] == tape_id:
                # Pass lists to converter, even though only one item
                converted = _chroma_result_to_tape(
                    ids=results["ids"],
                    metadatas=results["metadatas"],
                    documents=results["documents"],
                    embeddings=results["embeddings"],
                )
                return converted[0] if converted else None
            else:
                logger.debug(f"Tape {tape_id} not found in Chroma.")
                return None
        except Exception as e:
            # Catch potential index errors or other issues if ID not found
            logger.error(f"Failed to retrieve tape {tape_id} from Chroma: {e}", exc_info=True)
            return None

    def get_tapes_batch(self, tape_ids: List[str]) -> List[Optional[Tape]]:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")
        if not tape_ids:
            return []
        try:
            import numpy as np

            results = self.collection.get(ids=tape_ids, include=["metadatas", "documents", "embeddings"])

            # Convert embeddings to lists
            converted_embeddings: List[Optional[List[float]]] = []
            for emb in results.get("embeddings", []):
                if emb is None:
                    converted_embeddings.append(None)
                elif isinstance(emb, np.ndarray):
                    converted_embeddings.append(emb.tolist())
                    logger.debug(f"Converted NumPy array to list for embedding, length: {len(emb)}")
                elif isinstance(emb, (list, tuple)):
                    converted_embeddings.append(list(emb))
                    logger.debug(f"Embedding is already a list/tuple, length: {len(emb)}")
                else:
                    logger.warning(f"Unexpected embedding type: {type(emb).__name__}")
                    converted_embeddings.append(None)

            # Convert results, preserving order and handling missing items
            result_map = {res_id: i for i, res_id in enumerate(results["ids"])}  # map ID to index in results

            tapes = _chroma_result_to_tape(
                ids=results["ids"],
                metadatas=results["metadatas"],
                documents=results["documents"],
                embeddings=converted_embeddings,
            )

            tapes_map = {res_id: tape for res_id, tape in zip(results["ids"], tapes) if tape is not None}

            final_tapes: List[Optional[Tape]] = []
            for tid in tape_ids:
                if tid in tapes_map:
                    final_tapes.append(tapes_map[tid])
                else:
                    final_tapes.append(None)
                    logger.debug(f"Tape {tid} not found in Chroma during batch get.")
            logger.debug(f"Retrieved {sum(1 for t in final_tapes if t is not None)}/{len(tape_ids)} tapes from Chroma.")
            return final_tapes

        except Exception as e:
            logger.error(f"Failed to retrieve tape batch from Chroma: {e}", exc_info=True)
            return [None] * len(tape_ids)

    def update_tape(self, tape: Tape) -> None:
        # Chroma's upsert handles updates if ID exists
        logger.debug(f"Updating tape {tape.id} in Chroma using upsert.")
        self.add_tape(tape)

    def delete_tape(self, tape_id: str) -> bool:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")
        try:
            self.collection.delete(ids=[tape_id])
            # Chroma delete doesn't throw error if ID not found, just does nothing.
            logger.debug(f"Delete operation sent for tape {tape_id} in Chroma.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete tape {tape_id} from Chroma: {e}", exc_info=True)
            return False

    def vector_search(
        self, query_vector: List[float], top_k: int, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")

        chroma_where = _build_chroma_where_filter(filter_dict) if filter_dict else None
        logger.debug(f"Performing Chroma vector search: k={top_k}, where={chroma_where}")

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=chroma_where,
                include=["distances"],  # only need IDs and distances(scores)
            )
            # Results format: QueryResult(ids=[['id1', 'id2']], distances=[[0.1, 0.2]], ...)
            # Extract results for the first (only) query vector
            ids = results["ids"][0] if results and results["ids"] else []
            distances = results["distances"][0] if results and results["distances"] else []

            # Chroma returns distances, often want similarity (1 - distance for cosine)
            # Assuming cosine distance: similarity = 1 - distance
            # Check metadata if needed: distance_metric = self.collection.metadata.get("hnsw:space", "cosine")
            similarity_scores = [(1.0 - dist if dist is not None else 0.0) for dist in distances]

            return list(zip(ids, similarity_scores))

        except Exception as e:
            logger.error(f"Vector search failed in Chroma: {e}", exc_info=True)
            return []

    def get_all_tape_ids(self) -> List[str]:
        if not self.collection:
            raise ConnectionError("Chroma collection not initialized")
        logger.warning("get_all_tape_ids fetching all results from Chroma, may be slow.")
        try:
            results = self.collection.get(include=[])
            return cast(List[str], results.get("ids", []))
        except Exception as e:
            logger.error(f"Failed to get all tape IDs from Chroma: {e}", exc_info=True)
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
        Finds Tape IDs based solely on metadata filtering criteria in ChromaDB.

        Args:
            filter_dict: Dictionary defining metadata filters (e.g., {'status': 'active', 'priority_lte': 3}).
                         Supports operators via keys like 'priority_lte' or explicit {'field': {'$op': value}}.
            limit: Maximum number of IDs to return. If None, returns all matching IDs (subject to Chroma limits).
            offset: Number of IDs to skip for pagination. Defaults to 0 if None.
            sort_by: Metadata field to sort results by (e.g., 'created_at'). Sorting is performed in-memory.
                     If None, results are unsorted.
            sort_desc: If True, sort in descending order; otherwise, ascending.

        Returns:
            List[str]: A list of Tape IDs matching the filter criteria, potentially limited, offset, and sorted.
                       Returns an empty list if none found or on error (errors are logged).
        """
        if not self.collection:
            logger.error("Chroma collection not initialized")
            return []

        try:
            # Translate filter_dict to Chroma's 'where' format
            where_filter = _build_chroma_where_filter(filter_dict) if filter_dict else None
            logger.debug(
                f"Chroma find_ids_by_filter: where={where_filter}, limit={limit}, offset={offset}, sort_by={sort_by}, sort_desc={sort_desc}"
            )

            # Fetch IDs using Chroma's get method with where filter
            results = self.collection.get(
                where=where_filter,
                include=[],  # only fetch IDs to minimize data transfer
            )
            ids: List[str] = results.get("ids", [])
            logger.debug(f"Retrieved {len(ids)} IDs from Chroma for filter: {filter_dict}")

            # Apply sorting if requested (in-memory, as Chroma doesn't support native sorting)
            if sort_by:
                logger.warning(
                    f"Sorting by '{sort_by}' is performed in-memory, which may be inefficient for large result sets."
                )
                # To sort, we need metadata for the sort_by field
                results_with_meta = self.collection.get(
                    where=where_filter,
                    include=["metadatas"],
                )
                ids = results_with_meta.get("ids", [])
                metadatas = results_with_meta.get("metadatas", [])

                if not ids or not metadatas:
                    logger.debug("No results with metadata for sorting")
                    return []

                # Pair IDs with sort values
                id_value_pairs = []
                for id_, meta in zip(ids, metadatas):
                    if meta and sort_by in meta:
                        value = meta[sort_by]
                        if sort_by in ["created_at", "updated_at", "last_accessed", "expires_at"] and isinstance(
                            value, str
                        ):
                            try:
                                value = datetime.fromisoformat(value)
                            except ValueError:
                                logger.warning(f"Invalid datetime format for {sort_by} in tape {id_}: {value}")
                                continue
                        id_value_pairs.append((id_, value))
                    else:
                        logger.debug(f"Missing sort_by field '{sort_by}' in metadata for tape {id_}")
                        continue

                # Sort pairs based on value
                id_value_pairs.sort(key=lambda x: x[1], reverse=sort_desc)
                ids = [pair[0] for pair in id_value_pairs]

            # Apply pagination
            offset = offset or 0
            if offset > len(ids):
                logger.debug(f"Offset {offset} exceeds result count {len(ids)}")
                return []

            if limit is not None:
                ids = ids[offset : offset + limit]
            else:
                ids = ids[offset:]

            logger.debug(f"Returning {len(ids)} IDs after applying offset={offset} and limit={limit}")
            return ids

        except Exception as e:
            logger.error(f"Failed to find IDs by filter in Chroma: {e}", exc_info=True)
            return []

    def close(self) -> None:
        logger.info("ChromaAdapter close called (PersistentClient usually doesn't require explicit close).")
        # For HttpClient, we might need client.stop() here
        self.client = None
        self.collection = None
