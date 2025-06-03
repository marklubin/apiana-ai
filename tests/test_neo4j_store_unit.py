"""
Unit tests for Neo4jMemoryStore with mocked dependencies.

These tests don't require a running Neo4j instance.
"""
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from apiana.storage.neo4j_store import Neo4jMemoryStore
from apiana.types.common import Conversation, Message
from apiana.types.configuration import Neo4jConfig


@pytest.fixture
def mock_neo4j_config() -> Neo4jConfig:
    """Create a mock Neo4j configuration."""
    return Neo4jConfig(
        username="test_user",
        password="test_pass",
        host="localhost",
        port=7687,
        database="test_db"
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    """Create a sample conversation for testing."""
    return Conversation(
        title="Test Conversation",
        messages=[
            Message(id="1", role="user", content={"text": "Hello"}),
            Message(id="2", role="assistant", content={"text": "Hi"}),
        ],
        openai_conversation_id="test-123",
    )


class TestNeo4jMemoryStoreUnit:
    """Unit tests for Neo4jMemoryStore."""

    @patch('apiana.storage.neo4j_store.db')
    @patch('apiana.storage.neo4j_store.neomodel_config')
    def test_init_sets_database_url(self, mock_neomodel_config, mock_db, mock_neo4j_config):
        """Given config, should set database URL and install labels."""
        # Act
        Neo4jMemoryStore(mock_neo4j_config)
        
        # Assert
        expected_url = "bolt://test_user:test_pass@localhost:7687"
        assert mock_neomodel_config.DATABASE_URL == expected_url
        mock_db.install_all_labels.assert_called_once()

    @patch('apiana.storage.neo4j_store.Block')
    @patch('apiana.storage.neo4j_store.Grounding')
    @patch('apiana.storage.neo4j_store.Tag')
    @patch('apiana.storage.neo4j_store.datetime')
    @patch('apiana.storage.neo4j_store.db')
    @patch('apiana.storage.neo4j_store.neomodel_config')
    def test_store_convo_creates_block_with_relationships(
        self, mock_neomodel_config, mock_db, mock_datetime, 
        mock_tag_class, mock_grounding_class, mock_block_class,
        mock_neo4j_config, sample_conversation
    ):
        """Given conversation data, should create Block with proper relationships."""
        # Arrange
        test_time = datetime(2024, 1, 15, 10, 0, 0)
        mock_datetime.utcnow.return_value = test_time
        
        # Mock Tag instances
        mock_tag1 = Mock()
        mock_tag1.name = "tag1"
        mock_tag2 = Mock()
        mock_tag2.name = "tag2"
        
        # Mock Tag.nodes.filter to return empty (no existing tags)
        mock_tag_nodes = Mock()
        mock_tag_nodes.filter.return_value = []
        mock_tag_class.nodes = mock_tag_nodes
        mock_tag_class.return_value.save.side_effect = [mock_tag1, mock_tag2]
        
        # Mock Grounding
        mock_grounding = Mock()
        mock_grounding_class.get_or_create.return_value = [mock_grounding]
        
        # Mock Block
        mock_block = Mock()
        mock_block.tagged_with = Mock()
        mock_block.grounded_by = Mock()
        mock_block_class.return_value.save.return_value = mock_block
        
        # Create store and call method
        store = Neo4jMemoryStore(mock_neo4j_config)
        summary = "Test summary"
        embeddings = [0.1, 0.2, 0.3]
        tags = ["tag1", "tag2"]
        
        # Act
        result = store.store_convo(sample_conversation, summary, embeddings, tags)
        
        # Assert
        # Check Block was created with correct parameters
        mock_block_class.assert_called_once_with(
            content=summary,
            created_at=test_time,
            updated_at=test_time,
            embedding_v1=embeddings,
            tags={"tag1": mock_tag1, "tag2": mock_tag2},
            block_type="experience",
            experience_type="conversation",
        )
        
        # Check Grounding was created
        mock_grounding_class.get_or_create.assert_called_once_with({
            "external_id": "test-123",
            "external_label": "Test Conversation",
            "external_source": "conversation",
        })
        
        # Check relationships were connected
        mock_block.grounded_by.connect.assert_called_once_with(mock_grounding)
        assert mock_block.tagged_with.connect.call_count == 2
        
        # Check return value
        assert result == mock_block

    @patch('apiana.storage.neo4j_store.Block')
    @patch('apiana.storage.neo4j_store.Grounding')
    @patch('apiana.storage.neo4j_store.Tag')
    @patch('apiana.storage.neo4j_store.datetime')
    @patch('apiana.storage.neo4j_store.db')
    @patch('apiana.storage.neo4j_store.neomodel_config')
    def test_store_convo_reuses_existing_tags(
        self, mock_neomodel_config, mock_db, mock_datetime,
        mock_tag_class, mock_grounding_class, mock_block_class,
        mock_neo4j_config, sample_conversation
    ):
        """Given existing tags, should reuse them instead of creating new ones."""
        # Arrange
        test_time = datetime(2024, 1, 15, 10, 0, 0)
        mock_datetime.utcnow.return_value = test_time
        
        # Mock existing tags
        mock_existing_tag = Mock()
        mock_existing_tag.name = "existing"
        
        # Mock Tag.nodes.filter to return existing tag
        mock_tag_nodes = Mock()
        mock_tag_nodes.filter.side_effect = lambda name: [mock_existing_tag] if name == "existing" else []
        mock_tag_class.nodes = mock_tag_nodes
        
        # Mock new tag creation
        mock_new_tag = Mock()
        mock_new_tag.name = "new"
        mock_tag_class.return_value.save.return_value = mock_new_tag
        
        # Mock other dependencies
        mock_grounding_class.get_or_create.return_value = [Mock()]
        mock_block = Mock()
        mock_block.tagged_with = Mock()
        mock_block.grounded_by = Mock()
        mock_block_class.return_value.save.return_value = mock_block
        
        # Create store and call method
        store = Neo4jMemoryStore(mock_neo4j_config)
        
        # Act
        store.store_convo(
            sample_conversation, "Summary", [0.1], ["existing", "new"]
        )
        
        # Assert
        # Check that existing tag was not recreated
        assert mock_tag_class.call_count == 1  # Only called for "new" tag
        mock_tag_class.assert_called_with(name="new", created_at=test_time)
        
        # Check Block was created with both tags
        mock_block_class.assert_called_once()
        call_args = mock_block_class.call_args[1]
        assert "existing" in call_args["tags"]
        assert "new" in call_args["tags"]
        assert call_args["tags"]["existing"] == mock_existing_tag
        assert call_args["tags"]["new"] == mock_new_tag

    @patch('apiana.storage.neo4j_store.Block')
    @patch('apiana.storage.neo4j_store.Grounding')
    @patch('apiana.storage.neo4j_store.Tag')
    @patch('apiana.storage.neo4j_store.datetime')
    @patch('apiana.storage.neo4j_store.db')
    @patch('apiana.storage.neo4j_store.neomodel_config')
    def test_store_convo_handles_empty_conversation_id(
        self, mock_neomodel_config, mock_db, mock_datetime,
        mock_tag_class, mock_grounding_class, mock_block_class,
        mock_neo4j_config
    ):
        """Given conversation with empty ID, should still create grounding."""
        # Arrange
        mock_datetime.utcnow.return_value = datetime.utcnow()
        conversation = Conversation(
            title="No ID",
            messages=[],
            openai_conversation_id=""  # Empty ID
        )
        
        mock_tag_class.nodes.filter.return_value = []
        mock_grounding = Mock()
        mock_grounding_class.get_or_create.return_value = [mock_grounding]
        mock_block = Mock()
        mock_block.tagged_with = Mock()
        mock_block.grounded_by = Mock()
        mock_block_class.return_value.save.return_value = mock_block
        
        store = Neo4jMemoryStore(mock_neo4j_config)
        
        # Act
        store.store_convo(conversation, "Summary", [0.1], [])
        
        # Assert
        mock_grounding_class.get_or_create.assert_called_once_with({
            "external_id": "",
            "external_label": "No ID",
            "external_source": "conversation",
        })