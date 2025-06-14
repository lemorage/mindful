from __future__ import annotations  # allows forward refs

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from mindful.memory.tape import Tape  # Only for type hints


class StorageAdapter(ABC):
    """Abstract Base Class defining the interface for storing and retrieving Tapes."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def add_tape(self, tape: Tape) -> None:
        pass

    def add_tapes_batch(self, tapes: List[Tape]) -> None:
        for tape in tapes:
            self.add_tape(tape)

    @abstractmethod
    def get_tape(self, tape_id: str) -> Optional[Tape]:
        pass

    def get_tapes_batch(self, tape_ids: List[str]) -> List[Optional[Tape]]:
        return [self.get_tape(tid) for tid in tape_ids]

    @abstractmethod
    def update_tape(self, tape: Tape) -> None:
        pass

    @abstractmethod
    def delete_tape(self, tape_id: str) -> bool:
        """Deletes a Tape by its ID. Returns True if successful."""
        pass

    @abstractmethod
    def vector_search(
        self, query_vector: List[float], top_k: int, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Performs vector similarity search.

        Args:
            query_vector: The vector representation of the query.
            top_k: The number of top results to return.
            filter_dict: Optional dictionary for metadata filtering during search.
                         Structure depends on the backend implementation.

        Returns:
            A list of tuples, each containing (tape_id, similarity_score).
            Sorted by score descending.
        """
        pass

    @abstractmethod
    def get_all_tape_ids(self) -> List[str]:
        """Retrieves all tape IDs (needed for some evolution tasks, use cautiously)."""
        pass

    @abstractmethod
    def find_ids_by_filter(
        self,
        filter_dict: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = True,
    ) -> List[str]:
        """
        Finds Tape IDs based solely on metadata filtering criteria.

        Args:
            filter_dict (Dict[str, Any]): Dictionary defining the metadata filters.
                The structure and supported operators (e.g., {'priority_lte': 5},
                {'status': 'active'}, {'role': {'$in': ['user', 'assistant']}})
                depend on the concrete adapter's translation logic for its backend.
                Keys should correspond to indexed Tape metadata fields.
            limit (Optional[int]): Maximum number of IDs to return.
            offset (Optional[int]): Number of IDs to skip (for pagination).
            sort_by (Optional[str]): Metadata field to sort results by (e.g., 'last_accessed').
                                      Support depends on backend capability.
            sort_desc (bool): If True, sort descending; otherwise ascending.

        Returns:
            List[str]: A list of Tape IDs matching the filter criteria, potentially
                       limited, offset, and sorted. Returns empty list if none found
                       or on error (implementations should log errors).
        """
        pass

    # TODO: Add methods for keyword search, complex metadata queries if needed
    # @abstractmethod
    # def keyword_search(...) -> List[Tuple[str, float]]: ...

    # TODO: Add methods specific to evolution tasks if direct DB query is better
    # @abstractmethod
    # def find_stale_tapes(...) -> Iterator[Tape]: ...

    @abstractmethod
    def close(self) -> None:
        pass
