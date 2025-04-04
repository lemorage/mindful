from datetime import datetime
from typing import Dict, List, Union
from uuid import uuid4
from pydantic import BaseModel, Field


class Tape(BaseModel):
    """
    A structured memory tape for the LLM bot, inspired by the Slip-Card (Zettelkasten) method.

    This class represents an atomic unit of knowledge in the bot's memory system, allowing
    intelligent retrieval, dynamic linking, and evolutionary tracking of stored information.

    Attributes:
        content (str): The main textual information stored in this memory note.
        role (str): The role associated with this memory (e.g., 'user', 'assistant').
        uid (str): A structured unique identifier based on a timestamp (YYYYMMDDHHMMSS format).
        id (str): A globally unique identifier (UUID) for the note.
        category (str): A broad classification label for grouping similar notes.
        context (str): The situational or thematic context in which this memory is relevant.
        keywords (List[str]): A list of keywords for efficient indexing and retrieval.
        links (Dict[str, str]): A dictionary of related memory notes (bidirectional linking).
        related_queries (List[str]): A record of user queries that led to the creation of this note.
        embedding_vector (List[float]): A numerical vector representation for semantic retrieval.
        access_count (int): A counter tracking how often this note has been accessed.
        priority (int): A score (1-10) representing the importance of this note.
        created_at (datetime): The timestamp when this note was originally created.
        updated_at (datetime): The timestamp when this note was last modified.
        last_accessed (datetime): The timestamp when this note was last retrieved.
        versions (List[Dict[str, Union[str, datetime]]]): A history of previous versions of this note.
    """

    # Core tontent
    content: str = Field(..., description="Main textual content of the memory note.")
    role: str = Field(default="user", description="Role associated with this memory (e.g., 'user', 'assistant').")
    uid: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"), description="Time-based unique identifier."
    )
    id: str = Field(default_factory=lambda: str(uuid4()), description="Universally unique identifier for the note.")

    # Dynamic metadata (handled by LLM)
    category: str = Field(..., description="High-level classification of the note.")
    context: str = Field(..., description="Situational or thematic context of the note.")
    keywords: List[str] = Field(default_factory=list, description="List of keywords for retrieval and indexing.")
    links: Dict[str, str] = Field(default_factory=dict, description="Bidirectional links to related memory notes.")
    related_queries: List[str] = Field(
        default_factory=list, description="Similar or related queries linked to this note."
    )
    embedding_vector: List[float] = Field(None, description="Vector representation for semantic search and retrieval.")

    # Spatial tracking
    access_count: int = Field(0, ge=0, description="Number of times this note has been retrieved.")
    priority: int = Field(1, ge=1, le=10, description="Priority level (1-10) for importance ranking.")

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
