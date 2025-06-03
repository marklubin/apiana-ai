"""
Integration tests for Neo4jMemoryStore using a real Neo4j instance.

These tests require a running Neo4j instance (e.g., via docker-compose).
"""
from datetime import datetime
from unittest.mock import patch

import pytest
from neomodel import db

from apiana.storage.neo4j_store import Neo4jMemoryStore
from apiana.types.common import Conversation, Message
from apiana.types.configuration import Neo4jConfig


@pytest.fixture
def neo4j_config() -> Neo4jConfig:
    """Create Neo4j configuration for integration tests."""
    # Use the actual running Neo4j instance
    return Neo4jConfig(
        username="neo4j",
        password="password",
        host="localhost",
        port=7687,  # Standard Neo4j port
        database=None
    )


@pytest.fixture
def memory_store(neo4j_config) -> Neo4jMemoryStore:
    """Create a Neo4jMemoryStore instance for tests."""
    return Neo4jMemoryStore(neo4j_config)


@pytest.fixture
def sample_conversation() -> Conversation:
    """Create a sample conversation for testing."""
    return Conversation(
        title="Test Conversation",
        messages=[
            Message(
                id="1",
                role="user",
                content={"text": "Hello, how are you?"},
                created_at=datetime.utcnow(),
            ),
            Message(
                id="2",
                role="assistant",
                content={"text": "I'm doing well, thank you!"},
                created_at=datetime.utcnow(),
            ),
        ],
        openai_conversation_id="test-conv-123",
        create_time=datetime.utcnow(),
        update_time=datetime.utcnow(),
    )


@pytest.fixture
def sample_embeddings() -> list[float]:
    """Create sample embeddings for testing."""
    # Create a small vector for testing (normally would be 384+ dimensions)
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.integration
class TestNeo4jMemoryStoreIntegration:
    """Integration test suite for Neo4jMemoryStore."""

    def test_initialize_store_with_valid_config(self, neo4j_config):
        """Given valid config, should initialize and connect to Neo4j."""
        Neo4jMemoryStore(neo4j_config)
        
        # Verify that we can make a basic query
        result, _ = db.cypher_query("RETURN 1 as test")
        assert result[0][0] == 1

    def test_store_conversation_with_all_data(
        self, memory_store, sample_conversation, sample_embeddings
    ):
        """Given complete data, should create Block with all relationships."""
        # Arrange
        summary = "We had a brief greeting exchange."
        tags = ["greeting", "social", "friendly"]
        
        # Act
        block = memory_store.store_convo(
            sample_conversation, summary, sample_embeddings, tags
        )
        
        # Assert
        assert block is not None
        assert block.content == summary
        assert block.embedding_v1 == sample_embeddings
        assert block.block_type == "experience"
        assert block.experience_type == "conversation"
        
        # Verify tags were created
        assert len(list(block.tagged_with.all())) == 3
        tag_names = [tag.name for tag in block.tagged_with.all()]
        assert set(tag_names) == set(tags)
        
        # Verify grounding was created
        groundings = list(block.grounded_by.all())
        assert len(groundings) == 1
        grounding = groundings[0]
        assert grounding.external_id == "test-conv-123"
        assert grounding.external_label == "Test Conversation"
        assert grounding.external_source == "conversation"

    def test_store_conversation_with_empty_tags(
        self, memory_store, sample_conversation, sample_embeddings
    ):
        """Given no tags, should create Block without tag relationships."""
        # Arrange
        summary = "A conversation without tags."
        tags = []
        
        # Act
        block = memory_store.store_convo(
            sample_conversation, summary, sample_embeddings, tags
        )
        
        # Assert
        assert block is not None
        assert len(list(block.tagged_with.all())) == 0

    def test_store_conversation_with_duplicate_tags(
        self, memory_store, sample_conversation, sample_embeddings
    ):
        """Given duplicate tags, should only create unique tags."""
        # Arrange
        summary = "A conversation with duplicate tags."
        tags = ["test", "test", "duplicate", "duplicate", "unique"]
        
        # Act
        block = memory_store.store_convo(
            sample_conversation, summary, sample_embeddings, tags
        )
        
        # Assert
        tag_names = [tag.name for tag in block.tagged_with.all()]
        assert len(tag_names) == 3
        assert set(tag_names) == {"test", "duplicate", "unique"}

    def test_handle_missing_conversation_id(
        self, memory_store, sample_embeddings
    ):
        """Given conversation with no ID, should create grounding with empty ID."""
        # Arrange
        conversation = Conversation(
            title="No ID Conversation",
            messages=[],
            openai_conversation_id="",  # Empty ID
        )
        summary = "A conversation without an ID."
        
        # Act
        block = memory_store.store_convo(
            conversation, summary, sample_embeddings, []
        )
        
        # Assert
        groundings = list(block.grounded_by.all())
        assert len(groundings) == 1
        assert groundings[0].external_id == ""
        assert groundings[0].external_label == "No ID Conversation"

    def test_verify_datetime_handling(
        self, memory_store, sample_conversation, sample_embeddings
    ):
        """Given current time, should set created_at and updated_at correctly."""
        # Arrange
        summary = "Testing datetime fields."
        
        # Mock datetime to control the time
        test_time = datetime(2024, 1, 15, 10, 30, 0)
        with patch('apiana.storage.neo4j_store.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = test_time
            
            # Act
            block = memory_store.store_convo(
                sample_conversation, summary, sample_embeddings, []
            )
            
            # Assert
            assert block.created_at == test_time
            assert block.updated_at == test_time

    def test_reuse_existing_tags(
        self, memory_store, sample_conversation, sample_embeddings
    ):
        """Given existing tags, should reuse them instead of creating duplicates."""
        # Arrange
        summary1 = "First conversation"
        summary2 = "Second conversation"
        tags = ["reusable", "test"]
        
        # Act - Create first block
        block1 = memory_store.store_convo(
            sample_conversation, summary1, sample_embeddings, tags
        )
        
        # Modify conversation for second block
        conversation2 = Conversation(
            title="Second Test",
            messages=[],
            openai_conversation_id="test-conv-456",
        )
        
        # Create second block with same tags
        block2 = memory_store.store_convo(
            conversation2, summary2, sample_embeddings, tags
        )
        
        # Assert - Check that tags are reused
        tags1 = list(block1.tagged_with.all())
        tags2 = list(block2.tagged_with.all())
        
        # Same tag objects should be used
        assert len(tags1) == len(tags2) == 2
        tag_ids1 = {tag.element_id for tag in tags1}
        tag_ids2 = {tag.element_id for tag in tags2}
        assert tag_ids1 == tag_ids2

    def test_reuse_existing_grounding(
        self, memory_store, sample_conversation, sample_embeddings
    ):
        """Given existing grounding, should reuse it for same conversation ID."""
        # Arrange
        summary1 = "First summary of conversation"
        summary2 = "Second summary of same conversation"
        
        # Act - Create two blocks for same conversation
        block1 = memory_store.store_convo(
            sample_conversation, summary1, sample_embeddings, []
        )
        block2 = memory_store.store_convo(
            sample_conversation, summary2, sample_embeddings, []
        )
        
        # Assert - Same grounding should be used
        grounding1 = list(block1.grounded_by.all())[0]
        grounding2 = list(block2.grounded_by.all())[0]
        assert grounding1.element_id == grounding2.element_id

    def test_large_embedding_vector(
        self, memory_store, sample_conversation
    ):
        """Given large embedding vector, should store correctly."""
        # Arrange - Create a realistic 384-dimension vector
        large_embeddings = [0.1 * i for i in range(384)]
        summary = "Testing with large embeddings"
        
        # Act
        block = memory_store.store_convo(
            sample_conversation, summary, large_embeddings, []
        )
        
        # Assert
        assert len(block.embedding_v1) == 384
        assert block.embedding_v1 == large_embeddings

    @pytest.fixture(autouse=True)
    def cleanup_database(self, memory_store):
        """Clean up database after each test."""
        yield
        # Clean up all test nodes after each test
        try:
            db.cypher_query("""
                MATCH (n)
                WHERE n.block_type = 'experience' 
                   OR n.external_source = 'conversation'
                   OR n.name IN ['greeting', 'social', 'friendly', 'test', 'duplicate', 'unique', 'reusable']
                DETACH DELETE n
            """)
        except Exception:
            # If cleanup fails, it's okay - tests should still work
            pass