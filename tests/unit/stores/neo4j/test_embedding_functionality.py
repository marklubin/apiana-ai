"""Unit tests for embedding functionality in ApplicationStore."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from apiana.stores.neo4j.application_store import (
    ApplicationStore,
    create_embedding_hash
)
from apiana.types.chat_fragment import ChatFragment
from apiana.types.configuration import Neo4jConfig


class TestCreateEmbeddingHash:
    """Test cases for the create_embedding_hash function."""
    
    def test_basic_hash_creation(self):
        """Test basic hash creation."""
        hash_key = create_embedding_hash(
            "This is a test summary",
            "openai",
            "text-embedding-ada-002"
        )
        
        assert isinstance(hash_key, str)
        assert len(hash_key) == 32  # 32 character hash
    
    def test_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        text = "This is a test summary"
        provider = "openai"
        model = "text-embedding-ada-002"
        
        hash1 = create_embedding_hash(text, provider, model)
        hash2 = create_embedding_hash(text, provider, model)
        
        assert hash1 == hash2
    
    def test_hash_different_text(self):
        """Test that different text produces different hashes."""
        hash1 = create_embedding_hash(
            "First summary",
            "openai",
            "text-embedding-ada-002"
        )
        hash2 = create_embedding_hash(
            "Second summary",
            "openai",
            "text-embedding-ada-002"
        )
        
        assert hash1 != hash2
    
    def test_hash_different_provider(self):
        """Test that different providers produce different hashes."""
        text = "Same summary text"
        
        hash1 = create_embedding_hash(text, "openai", "model")
        hash2 = create_embedding_hash(text, "cohere", "model")
        
        assert hash1 != hash2
    
    def test_hash_different_model(self):
        """Test that different models produce different hashes."""
        text = "Same summary text"
        
        hash1 = create_embedding_hash(text, "openai", "text-embedding-ada-002")
        hash2 = create_embedding_hash(text, "openai", "text-embedding-3-small")
        
        assert hash1 != hash2
    
    def test_hash_case_insensitive(self):
        """Test that provider and model names are case-insensitive."""
        text = "Test summary"
        
        hash1 = create_embedding_hash(text, "OpenAI", "Text-Embedding-Ada-002")
        hash2 = create_embedding_hash(text, "openai", "text-embedding-ada-002")
        
        assert hash1 == hash2
    
    def test_hash_empty_text(self):
        """Test hash with empty text."""
        hash_key = create_embedding_hash("", "openai", "model")
        
        assert isinstance(hash_key, str)
        assert len(hash_key) == 32
    
    def test_hash_unicode_text(self):
        """Test hash with unicode text."""
        text = "Hello 世界 🌍"
        hash_key = create_embedding_hash(text, "openai", "model")
        
        assert isinstance(hash_key, str)
        assert len(hash_key) == 32


class TestApplicationStoreEmbeddings:
    """Test cases for embedding functionality in ApplicationStore."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Neo4j configuration."""
        config = Mock(spec=Neo4jConfig)
        config.host = "localhost"
        config.port = "7687"
        config.username = "neo4j"
        config.password = "password"
        return config
    
    @pytest.fixture
    @patch('apiana.stores.neo4j.application_store.db')
    @patch('apiana.stores.neo4j.application_store.neomodel_config')
    def store(self, mock_neomodel_config, mock_db, mock_config):
        """Create ApplicationStore instance with mocked dependencies."""
        return ApplicationStore(mock_config)
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_store_embedding_new(self, mock_embedding_node_class, store):
        """Test storing a new embedding."""
        # Mock the nodes.filter().first() to return None (not found)
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None
        mock_embedding_node_class.nodes = mock_nodes
        
        # Mock the save method
        mock_embedding_instance = Mock()
        mock_embedding_node_class.return_value = mock_embedding_instance
        mock_embedding_instance.save.return_value = mock_embedding_instance
        
        # Mock ChatFragmentNode for relationship
        with patch('apiana.stores.neo4j.application_store.ChatFragmentNode') as mock_fragment_class:
            mock_fragment_nodes = Mock()
            mock_fragment_nodes.filter.return_value.first.return_value = None
            mock_fragment_class.nodes = mock_fragment_nodes
            
            # Store embedding
            store.store_embedding(
                source_id="frag-123",
                source_type="summary",
                source_text="Test summary",
                embedding_vector=[0.1, 0.2, 0.3],
                model_provider="openai",
                model_name="text-embedding-ada-002",
                metadata={"tokens": 10}
            )
        
        # Verify creation
        mock_embedding_node_class.assert_called_once()
        call_kwargs = mock_embedding_node_class.call_args[1]
        assert call_kwargs["source_id"] == "frag-123"
        assert call_kwargs["source_type"] == "summary"
        assert call_kwargs["vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["model_name"] == "text-embedding-ada-002"
        assert call_kwargs["model_dimensions"] == 3
        assert call_kwargs["metadata"] == {"tokens": 10}
        
        mock_embedding_instance.save.assert_called_once()
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_store_embedding_existing(self, mock_embedding_node_class, store):
        """Test updating an existing embedding."""
        # Mock existing embedding
        mock_existing = Mock()
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_existing
        mock_embedding_node_class.nodes = mock_nodes
        
        # Store embedding (should update existing)
        store.store_embedding(
            source_id="frag-123",
            source_type="summary",
            source_text="Test summary",
            embedding_vector=[0.4, 0.5, 0.6],
            model_provider="openai",
            model_name="text-embedding-ada-002"
        )
        
        # Verify update
        assert mock_existing.vector == [0.4, 0.5, 0.6]
        assert mock_existing.metadata == {}
        mock_existing.save.assert_called_once()
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    @patch('apiana.stores.neo4j.application_store.ChatFragmentNode')
    def test_store_embedding_with_fragment_link(
        self, mock_fragment_class, mock_embedding_node_class, store
    ):
        """Test storing embedding with fragment relationship."""
        # Mock embedding creation
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None
        mock_embedding_node_class.nodes = mock_nodes
        
        mock_embedding_instance = Mock()
        mock_embedding_node_class.return_value = mock_embedding_instance
        mock_embedding_instance.save.return_value = mock_embedding_instance
        
        # Mock existing fragment
        mock_fragment = Mock()
        mock_fragment.embeddings = Mock()
        mock_fragment_nodes = Mock()
        mock_fragment_nodes.filter.return_value.first.return_value = mock_fragment
        mock_fragment_class.nodes = mock_fragment_nodes
        
        # Store embedding
        store.store_embedding(
            source_id="frag-123",
            source_type="summary",
            source_text="Test summary",
            embedding_vector=[0.1, 0.2, 0.3],
            model_provider="openai",
            model_name="text-embedding-ada-002"
        )
        
        # Verify fragment relationship created
        mock_fragment.embeddings.connect.assert_called_once_with(mock_embedding_instance)
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_get_embedding_found(self, mock_embedding_node_class, store):
        """Test retrieving an existing embedding."""
        # Mock embedding node
        mock_node = Mock()
        mock_node.embedding_id = "frag-123:hash123"
        mock_node.vector = [0.1, 0.2, 0.3]
        mock_node.model_provider = "openai"
        mock_node.model_name = "text-embedding-ada-002"
        mock_node.model_dimensions = 3
        mock_node.source_hash = "hash123"
        mock_node.created_at = datetime(2024, 1, 1)
        mock_node.metadata = {"tokens": 10}
        
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_node
        mock_embedding_node_class.nodes = mock_nodes
        
        # Get embedding
        result = store.get_embedding(
            source_id="frag-123",
            model_provider="openai",
            model_name="text-embedding-ada-002"
        )
        
        assert result is not None
        assert result["embedding_id"] == "frag-123:hash123"
        assert result["vector"] == [0.1, 0.2, 0.3]
        assert result["model_provider"] == "openai"
        assert result["model_name"] == "text-embedding-ada-002"
        assert result["model_dimensions"] == 3
        assert result["metadata"] == {"tokens": 10}
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_get_embedding_not_found(self, mock_embedding_node_class, store):
        """Test retrieving non-existent embedding."""
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None
        mock_embedding_node_class.nodes = mock_nodes
        
        result = store.get_embedding(
            source_id="frag-123",
            model_provider="openai",
            model_name="text-embedding-ada-002"
        )
        
        assert result is None
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    @patch('apiana.stores.neo4j.application_store.create_embedding_hash')
    def test_get_embedding_with_source_text(
        self, mock_create_hash, mock_embedding_node_class, store
    ):
        """Test retrieving embedding with source text verification."""
        mock_create_hash.return_value = "testhash123"
        
        # Mock query chain
        mock_query = Mock()
        mock_nodes = Mock()
        mock_nodes.filter.return_value = mock_query
        mock_query.filter.return_value.first.return_value = None
        mock_embedding_node_class.nodes = mock_nodes
        
        # Get embedding with source text
        store.get_embedding(
            source_id="frag-123",
            model_provider="openai",
            model_name="text-embedding-ada-002",
            source_text="Test summary"
        )
        
        # Verify hash was created and used in filter
        mock_create_hash.assert_called_once_with(
            "Test summary", "openai", "text-embedding-ada-002"
        )
        mock_query.filter.assert_called_once_with(source_hash="testhash123")
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_list_embeddings_no_filters(self, mock_embedding_node_class, store):
        """Test listing embeddings without filters."""
        # Mock nodes
        mock_node1 = Mock()
        mock_node1.embedding_id = "id1"
        mock_node1.source_id = "frag1"
        mock_node1.source_type = "summary"
        mock_node1.model_provider = "openai"
        mock_node1.model_name = "model1"
        mock_node1.model_dimensions = 1536
        mock_node1.created_at = datetime(2024, 1, 1)
        mock_node1.metadata = {}
        
        mock_query = Mock()
        mock_slice_result = Mock()
        mock_slice_result.__iter__ = Mock(return_value=iter([mock_node1]))
        mock_query.order_by.return_value.__getitem__ = Mock(return_value=mock_slice_result)
        mock_embedding_node_class.nodes = mock_query
        
        # List embeddings
        results = store.list_embeddings()
        
        assert len(results) == 1
        assert results[0]["embedding_id"] == "id1"
        assert results[0]["source_id"] == "frag1"
        mock_query.order_by.assert_called_once_with('-created_at')
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_list_embeddings_with_filters(self, mock_embedding_node_class, store):
        """Test listing embeddings with filters."""
        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_slice_result = Mock()
        mock_slice_result.__iter__ = Mock(return_value=iter([]))
        mock_query.order_by.return_value.__getitem__ = Mock(return_value=mock_slice_result)
        mock_embedding_node_class.nodes = mock_query
        
        # List with filters
        store.list_embeddings(
            source_id="frag-123",
            source_type="summary",
            model_provider="openai",
            limit=10
        )
        
        # Verify filters applied
        filter_calls = mock_query.filter.call_args_list
        assert any(call[1] == {'source_id': 'frag-123'} for call in filter_calls)
        assert any(call[1] == {'source_type': 'summary'} for call in filter_calls)
        assert any(call[1] == {'model_provider': 'openai'} for call in filter_calls)


class TestApplicationStoreSummaryTracking:
    """Test cases for summary tracking functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Neo4j configuration."""
        config = Mock(spec=Neo4jConfig)
        config.host = "localhost"
        config.port = "7687"
        config.username = "neo4j"
        config.password = "password"
        return config
    
    @pytest.fixture
    @patch('apiana.stores.neo4j.application_store.db')
    @patch('apiana.stores.neo4j.application_store.neomodel_config')
    def store(self, mock_neomodel_config, mock_db, mock_config):
        """Create ApplicationStore instance with mocked dependencies."""
        return ApplicationStore(mock_config)
    
    @patch('apiana.stores.neo4j.application_store.ChatFragmentNode')
    def test_store_fragment_with_summary_info(self, mock_fragment_class, store):
        """Test storing fragment with summary agent ID and hash."""
        # Mock nodes.filter().first() to return None (new fragment)
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = None
        mock_fragment_class.nodes = mock_nodes
        
        # Mock save
        mock_fragment_instance = Mock()
        mock_fragment_class.return_value = mock_fragment_instance
        mock_fragment_instance.save.return_value = mock_fragment_instance
        
        # Create fragment
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        # Store with summary info
        store.store_fragment(
            fragment,
            summary_agent_id="agent-456",
            summary_text="This is a test summary"
        )
        
        # Verify creation with summary fields
        call_kwargs = mock_fragment_class.call_args[1]
        assert call_kwargs["summary_agent_id"] == "agent-456"
        assert call_kwargs["summary_hash"] is not None
        assert len(call_kwargs["summary_hash"]) == 32
    
    @patch('apiana.stores.neo4j.application_store.ChatFragmentNode')
    def test_update_fragment_with_summary_info(self, mock_fragment_class, store):
        """Test updating existing fragment with summary info."""
        # Mock existing fragment
        mock_existing = Mock()
        mock_existing.tags = ["existing_tag"]
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_existing
        mock_fragment_class.nodes = mock_nodes
        
        # Create fragment
        fragment = ChatFragment(
            fragment_id="frag-123",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        # Update with summary info
        store.store_fragment(
            fragment,
            summary_agent_id="agent-789",
            summary_text="Updated summary"
        )
        
        # Verify update
        assert mock_existing.summary_agent_id == "agent-789"
        assert hasattr(mock_existing, 'summary_hash')
        mock_existing.save.assert_called_once()
    
    @patch('apiana.stores.neo4j.application_store.ChatFragmentNode')
    def test_list_fragments_by_summary_agent(self, mock_fragment_class, store):
        """Test filtering fragments by summary agent ID."""
        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        # Return empty iterable
        mock_query.__iter__ = Mock(return_value=iter([]))
        mock_query.__getitem__ = Mock(return_value=mock_query)
        mock_fragment_class.nodes = mock_query
        
        # List with summary agent filter
        store.list_fragments(summary_agent_id="agent-123")
        
        # Verify filter applied
        mock_query.filter.assert_called_with(summary_agent_id="agent-123")
    
    @patch('apiana.stores.neo4j.application_store.ChatFragmentNode')
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')  
    def test_delete_fragment_with_embeddings(
        self, mock_embedding_class, mock_fragment_class, store
    ):
        """Test deleting fragment also deletes associated embeddings."""
        # Mock fragment with embeddings
        mock_fragment = Mock()
        mock_embedding1 = Mock()
        mock_embedding2 = Mock()
        mock_fragment.embeddings.all.return_value = [mock_embedding1, mock_embedding2]
        
        mock_nodes = Mock()
        mock_nodes.filter.return_value.first.return_value = mock_fragment
        mock_fragment_class.nodes = mock_nodes
        
        # Delete fragment
        result = store.delete_fragment("frag-123")
        
        assert result is True
        # Verify embeddings deleted
        mock_embedding1.delete.assert_called_once()
        mock_embedding2.delete.assert_called_once()
        # Verify fragment deleted
        mock_fragment.delete.assert_called_once()
    
    @patch('apiana.stores.neo4j.application_store.EmbeddingNode')
    def test_get_run_statistics_with_embeddings(self, mock_embedding_class, store):
        """Test statistics include embedding count."""
        # Mock node counts
        with patch('apiana.stores.neo4j.application_store.ChatFragmentNode') as mock_fragment_class:
            with patch('apiana.stores.neo4j.application_store.PipelineRunNode') as mock_run_class:
                mock_fragment_class.nodes.__len__.return_value = 100
                mock_run_class.nodes.__len__.return_value = 10
                mock_embedding_class.nodes.__len__.return_value = 500
                
                # Mock recent fragments query
                mock_query = Mock()
                mock_query.__len__ = Mock(return_value=25)
                mock_fragment_class.nodes.filter.return_value = mock_query
                
                # Get statistics
                stats = store.get_run_statistics()
                
                assert stats["total_fragments"] == 100
                assert stats["total_pipeline_runs"] == 10
                assert stats["total_embeddings"] == 500
                assert stats["fragments_added_today"] == 25