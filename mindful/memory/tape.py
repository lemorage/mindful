from datetime import datetime, timezone
import math
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4

import numpy as np
from pydantic import (
    BaseModel,
    Field,
)

from mindful.agent import MindfulAgent
from mindful.models import TapeMetadata

logger = logging.getLogger("mindful")


class Tape(BaseModel):
    """
    A structured memory tape for the LLM bot, inspired by the Slip-Card (Zettelkasten) method.

    This model represents an atomic unit of long-term memory, encapsulating knowledge that can
    be intelligently retrieved, contextually linked, and semantically evolved over time.

    Attributes:
        content (str): The main textual information stored in this memory note.
        role (str): The role associated with this memory (e.g., 'user', 'assistant').
        uid (str): A structured time-based identifier (YYYYMMDDHHMMSS format) for chronological sorting.
        id (str): A globally unique identifier (UUID) for the note instance.
        metadata (TapeMetadata): Encapsulated semantic metadata, including category, context, and keywords.
        embedding_vector (List[float]): A dense vector representation used for semantic search and retrieval.
        access_count (int): Tracks how many times this note has been accessed or retrieved.
        priority (int): Importance ranking from 1 (low) to 10 (critical).
        links (Dict[str, str]): Bidirectional links to other memory notes (e.g., related ideas, references).
        related_queries (List[str]): User queries or prompts that resulted in the creation of this note.
        created_at (datetime): Timestamp when the note was originally created.
        updated_at (datetime): Timestamp when the note was last updated.
        last_accessed (datetime): Timestamp of the most recent access.
        versions (List[Dict[str, Union[str, datetime]]]): Historical snapshots of previous versions of this note.
    """

    # Core content
    content: str = Field(..., description="Main textual content of the memory note.")
    role: str = Field(..., description="Role associated with this memory (e.g., 'user', 'assistant').")
    uid: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"), description="Time-based unique identifier."
    )
    id: str = Field(default_factory=lambda: str(uuid4()), description="Universally unique identifier for the note.")

    # Dynamic metadata (handled by mindful agent)
    metadata: TapeMetadata
    embedding_vector: List[float] = Field(..., description="Vector representation for semantic search and retrieval.")

    # Spatial tracking
    access_count: int = Field(0, ge=0, description="Number of times this note has been retrieved.")
    priority: int = Field(1, ge=1, le=10, description="Priority level (1-10) for importance ranking.")
    links: Dict[str, str] = Field(default_factory=dict, description="Bidirectional links to related memory notes.")
    related_queries: List[str] = Field(
        default_factory=list, description="Similar or related queries linked to this note."
    )

    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the note was created.")
    updated_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the note was last modified.")
    last_accessed: datetime = Field(
        default_factory=datetime.now, description="Timestamp when the note was last accessed."
    )

    # Evolutionary tracking
    versions: List[Dict[str, Union[str, datetime]]] = Field(
        default_factory=list, description="History of modifications and updates."
    )

    def update_content(self, new_content: str) -> None:
        """
        Update the content of the note while preserving the version history.

        This method appends the previous content to the `versions` list before updating the
        note's content and timestamp.

        Args:
            new_content (str): The new content to replace the existing content.
        """
        self.versions.append({"content": self.content, "timestamp": self.updated_at})
        self.content = new_content
        self.updated_at = datetime.now()

    def add_link(self, related_tape_id: str, description: str) -> None:
        """
        Establish a bidirectional link between this note and another.

        Args:
            related_tape_id (str): The unique identifier of the related note.
            description (str): A brief explanation of the relationship between the notes.
        """
        self.links[related_tape_id] = description


class TapeDeck:
    """A memory management system for storing and retrieving Tape objects."""

    def __init__(self, provider_name: str) -> None:
        self.tapes: Dict[str, Tape] = {}
        self.agent = MindfulAgent(provider_name)

    def add_tape(self, content: str, role: str) -> Tape:
        """
        Add a new Tape to the deck from raw content and role.
        Automatically handles embedding and placeholder metadata.

        Returns:
            Tape: The created and stored Tape object.
        """
        # Stub/default metadata
        category, context, keywords = self.agent.generate_metadata(content)
        category, context = (category or ""), (context or "")
        embedding_vector = self.agent.embed(content) or []

        # TODO: can later be executed through tool calling by an agent, empty for now
        links: Dict[str, str] = {}
        related_queries: List[str] = []

        # Create the Tape
        tape = Tape(
            content=content,
            role=role,
            metadata=TapeMetadata(category=category, context=context, keywords=keywords),
            links=links,
            related_queries=related_queries,
            embedding_vector=embedding_vector,
        )

        self.tapes[tape.id] = tape
        return tape

    def get_tape(self, tape_id: str) -> Optional[Tape]:
        """Retrieve a Tape by ID and update access metadata."""
        tape = self.tapes.get(tape_id)
        if tape:
            tape.access_count += 1
            tape.last_accessed = datetime.now()
        return tape

    def update_tape(self, tape_id: str, new_content: str) -> None:
        """Update a Tape's content and embedding."""
        tape = self.get_tape(tape_id)
        if tape:
            tape.update_content(new_content)
            tape.embedding_vector = self.agent.embed(new_content) or []

    def delete_tape(self, tape_id: str) -> None:
        """Delete a Tape by ID."""
        if tape_id in self.tapes:
            del self.tapes[tape_id]

    def link_tapes(self, tape_id1: str, tape_id2: str, description: str) -> None:
        """Create a bidirectional link between two Tapes."""
        tape1 = self.get_tape(tape_id1)
        tape2 = self.get_tape(tape_id2)
        if tape1 and tape2:
            tape1.add_link(tape_id2, description)
            tape2.add_link(tape_id1, f"replied_by_{description}")

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Tape]:
        """Retrieve relevant Tapes using embeddings if available, otherwise keywords."""
        if any(t.embedding_vector for t in self.tapes.values()):
            query_embedding = self.agent.embed(query)
            scores = {}
            for tape in self.tapes.values():
                if tape.embedding_vector:
                    # Cosine similarity
                    dot_product = np.dot(query_embedding, tape.embedding_vector)
                    norm = np.linalg.norm(query_embedding) * np.linalg.norm(tape.embedding_vector)
                    similarity = dot_product / norm if norm > 0 else 0

                    # priority 10 = 1.5x, priority 1 = 1.0x
                    weighted_score = similarity * (1 + (tape.priority - 1) * 0.05)
                    scores[tape.id] = weighted_score
            sorted_tapes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [self.tapes[tid] for tid, _ in sorted_tapes]
        else:
            # Fallback to keyword matching
            keywords = set(query.lower().split())
            scores = {}
            for tape in self.tapes.values():
                tape_keywords = set(tape.metadata.keywords)
                score = len(keywords.intersection(tape_keywords))
                if score > 0:
                    weighted_score = score * (1 + (tape.priority - 1) * 0.05)
                    scores[tape.id] = weighted_score
            sorted_tapes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [self.tapes[tid] for tid, _ in sorted_tapes]
