import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from mindful.memory.tape import Tape
from mindful.vector_store.chroma import ChromaAdapter, _tape_to_chroma_meta_and_doc


@pytest.fixture
def sample_tape():
    return Tape(
        id="tape1",
        content="Test content",
        embedding_vector=[0.1] * 128,
        role="user",
        status="active",
        priority=5,
        created_at=datetime(2023, 1, 1),
        keywords=["test", "example"],
    )


@pytest.fixture
def sample_tapes():
    return [
        Tape(
            id=f"tape{i}",
            content=f"Content {i}",
            embedding_vector=[0.1 * i] * 128,
            role="user",
            status="active" if i % 2 == 0 else "inactive",
            priority=i,
            created_at=datetime(2023, 1, i),
            keywords=[f"keyword{i}"],
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def chroma_adapter():
    adapter = ChromaAdapter()
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.get_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection
    adapter.client = mock_client
    adapter.collection = mock_collection
    return adapter


def test_initialize_success(chroma_adapter):
    # Arrange
    config = {"path": "./test_chroma_db", "collection_name": "test_collection", "create_collection_if_not_exists": True}
    with patch("chromadb.PersistentClient") as mock_client:
        mock_client.return_value = chroma_adapter.client

        # Act
        chroma_adapter.initialize(config)

        # Assert
        mock_client.assert_called_with(path=config["path"], tenant="default_tenant", database="default_database")
        chroma_adapter.client.get_collection.assert_called_with(name="test_collection")
        assert chroma_adapter.collection is not None


def test_initialize_collection_creation(chroma_adapter):
    # Arrange
    config = {"path": "./test_chroma_db", "collection_name": "test_collection", "create_collection_if_not_exists": True}
    chroma_adapter.client.get_collection.side_effect = Exception("Collection not found")
    with patch("chromadb.PersistentClient") as mock_client:
        mock_client.return_value = chroma_adapter.client

        # Act
        chroma_adapter.initialize(config)

        # Assert
        chroma_adapter.client.create_collection.assert_called_with(
            name="test_collection", metadata={"hnsw:space": "cosine"}
        )


def test_add_tape_success(chroma_adapter, sample_tape):
    # Arrange
    doc, meta = _tape_to_chroma_meta_and_doc(sample_tape)

    # Act
    chroma_adapter.add_tape(sample_tape)

    # Assert
    chroma_adapter.collection.upsert.assert_called_with(
        ids=[sample_tape.id], embeddings=[sample_tape.embedding_vector], documents=[doc], metadatas=[meta]
    )


def test_add_tape_missing_embedding(chroma_adapter, sample_tape):
    # Arrange
    sample_tape.embedding_vector = None

    # Act & Assert
    with pytest.raises(ValueError, match="Tape must have an embedding vector"):
        chroma_adapter.add_tape(sample_tape)


def test_add_tapes_batch_success(chroma_adapter, sample_tapes):
    # Arrange
    vectors = []
    for tape in sample_tapes:
        doc, meta = _tape_to_chroma_meta_and_doc(tape)
        vectors.append((tape.id, tape.embedding_vector, doc, meta))

    # Act
    chroma_adapter.add_tapes_batch(sample_tapes)

    # Assert
    chroma_adapter.collection.upsert.assert_called_with(
        ids=[tape.id for tape in sample_tapes],
        embeddings=[tape.embedding_vector for tape in sample_tapes],
        documents=[v[2] for v in vectors],
        metadatas=[v[3] for v in vectors],
    )


def test_get_tape_success(chroma_adapter, sample_tape):
    # Arrange
    doc, meta = _tape_to_chroma_meta_and_doc(sample_tape)
    chroma_adapter.collection.get.return_value = {
        "ids": [sample_tape.id],
        "metadatas": [meta],
        "documents": [doc],
        "embeddings": [sample_tape.embedding_vector],
    }

    # Act
    result = chroma_adapter.get_tape(sample_tape.id)

    # Assert
    assert result is not None
    assert result.id == sample_tape.id
    assert result.content == sample_tape.content
    assert result.status == sample_tape.status


def test_get_tape_not_found(chroma_adapter):
    # Arrange
    chroma_adapter.collection.get.return_value = {"ids": []}

    # Act
    result = chroma_adapter.get_tape("nonexistent")

    # Assert
    assert result is None


def test_find_ids_by_filter_status(chroma_adapter):
    # Arrange
    filter_dict = {"status": {"$eq": "active"}}
    expected_ids = ["tape1", "tape3"]
    chroma_adapter.collection.get.side_effect = [
        {"ids": expected_ids, "metadatas": [{"status": "active"}] * len(expected_ids)},
        {"ids": [], "metadatas": []},
    ]

    # Act
    result = chroma_adapter.find_ids_by_filter(filter_dict)

    # Assert
    assert result == expected_ids
    chroma_adapter.collection.get.assert_called_with(
        where={"status": {"$eq": "active"}}, limit=1000, offset=2, include=[]
    )


def test_find_ids_by_filter_pagination(chroma_adapter):
    # Arrange
    filter_dict = {"priority_gte": 2}
    all_ids = ["tape1", "tape2", "tape3", "tape4"]
    chroma_adapter.collection.get.return_value = {
        "ids": all_ids[1:3],
        "metadatas": [{"priority": i} for i in range(1, 5)],
    }

    # Act
    result = chroma_adapter.find_ids_by_filter(filter_dict, limit=2, offset=1)

    # Assert
    assert result == ["tape2", "tape3"]
    chroma_adapter.collection.get.assert_called_with(where={"priority": {"$gte": 2}}, limit=2, offset=1, include=[])


def test_find_ids_by_filter_sorting(chroma_adapter):
    # Arrange
    filter_dict = {"status": {"$eq": "active"}}
    all_ids = ["tape1", "tape2", "tape3"]
    metadatas = [
        {"status": "active", "created_at": "2023-01-01T00:00:00"},
        {"status": "active", "created_at": "2023-01-03T00:00:00"},
        {"status": "active", "created_at": "2023-01-02T00:00:00"},
    ]
    chroma_adapter.collection.get.side_effect = [{"ids": all_ids, "metadatas": metadatas}, {"ids": [], "metadatas": []}]

    # Act
    result = chroma_adapter.find_ids_by_filter(filter_dict, sort_by="created_at", sort_desc=False)  # ascending

    # Assert
    assert result == ["tape1", "tape3", "tape2"]  # sorted ascending by created_at


def test_find_ids_by_filter_empty(chroma_adapter):
    # Arrange
    filter_dict = {"status": {"$eq": "nonexistent"}}
    chroma_adapter.collection.get.return_value = {"ids": []}

    # Act
    result = chroma_adapter.find_ids_by_filter(filter_dict)

    # Assert
    assert result == []


def test_vector_search_success(chroma_adapter):
    # Arrange
    query_vector = [0.1] * 128
    top_k = 2
    filter_dict = {"role": "user"}
    expected_results = [("tape1", 0.9), ("tape2", 0.8)]
    chroma_adapter.collection.query.return_value = {
        "ids": [[id_ for id_, _ in expected_results]],
        "distances": [[1.0 - score for _, score in expected_results]],
    }

    # Act
    result = chroma_adapter.vector_search(query_vector, top_k, filter_dict)

    # Assert
    assert result == expected_results
    chroma_adapter.collection.query.assert_called_with(
        query_embeddings=[query_vector], n_results=top_k, where={"role": {"$eq": "user"}}, include=["distances"]
    )


def test_delete_tape_success(chroma_adapter):
    # Arrange
    tape_id = "tape1"
    chroma_adapter.collection.delete.return_value = None

    # Act
    result = chroma_adapter.delete_tape(tape_id)

    # Assert
    assert result is True
    chroma_adapter.collection.delete.assert_called_with(ids=[tape_id])
