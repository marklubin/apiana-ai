from datetime import datetime
from typing import List

from neomodel import config as neomodel_config
from neomodel import db

from apiana.types.configuration import Neo4jConfig
from apiana.types.memory_block import Block, Tag, Grounding
from apiana.types.chat_fragment import ChatFragment


class AgentMemoryStore:
    """
    Memory store for agent-specific data using the single Neo4j database.
    
    This store automatically tags all memory blocks with the agent_id,
    allowing logical separation of agent memories within the shared database.
    Works with Neo4j Community Edition.
    """
    
    def __init__(self, config: Neo4jConfig, agent_id: str):
        """
        Initialize the memory store for a specific agent.
        
        Args:
            config: Neo4j connection configuration
            agent_id: Unique identifier for the agent (automatically applied to all memories)
        """
        self.agent_id = agent_id
        self.config = config
        
        # Configure connection to the default database
        config_url = (
            f"bolt://{config.username}:{config.password}@{config.host}:{config.port}"
        )
        
        # Always use the default 'neo4j' database (Community Edition)
        neomodel_config.DATABASE_URL = f"{config_url}/neo4j"
        
        # Install labels for the shared database
        db.install_all_labels()
    
    def get_database_name(self) -> str:
        """Get the database name (always 'neo4j' for Community Edition)."""
        return "neo4j"
    
    def get_agent_id(self) -> str:
        """Get the agent identifier used for this store."""
        return self.agent_id

    def store_fragment(
        self,
        fragment: ChatFragment,
        summary: str,
        embeddings: List[float],
        tags: List[str],
    ) -> Block:
        """Store a ChatFragment in Neo4j."""
        # Get or create Tag nodes
        now = datetime.utcnow()
        db_tags = {}
        for tag_name in tags:
            # Try to get existing tag first
            existing_tags = Tag.nodes.filter(name=tag_name)
            if existing_tags:
                db_tags[tag_name] = existing_tags[0]
            else:
                # Create new tag with timestamp
                db_tags[tag_name] = Tag(name=tag_name, created_at=now).save()

        grounding = Grounding.get_or_create(
            {
                "external_id": fragment.openai_conversation_id or fragment.fragment_id,
                "external_label": fragment.title,
                "external_source": "chat_fragment",
            }
        )[0]

        # Create a parent Block for the fragment with agent_id automatically set
        block = Block(
            content=summary,
            created_at=fragment.create_time,
            updated_at=fragment.update_time,
            embedding_v1=embeddings,
            tags=db_tags,
            block_type="experience",
            experience_type="conversation",
            agent_id=self.agent_id,  # Automatically apply agent_id
        ).save()

        block.grounded_by.connect(grounding)
        [block.tagged_with.connect(tag) for tag in db_tags.values()]
        return block
    
    def get_memories(self, limit: int = 10, skip: int = 0) -> List[Block]:
        """
        Retrieve memories for this specific agent.
        
        Args:
            limit: Maximum number of memories to return
            skip: Number of memories to skip (for pagination)
            
        Returns:
            List of Block instances for this agent
        """
        # Query only blocks belonging to this agent
        return Block.nodes.filter(agent_id=self.agent_id).order_by('-created_at')[skip:skip+limit]
    
    def search_memories(self, embedding: List[float], threshold: float = 0.7, limit: int = 10) -> List[Block]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity score
            limit: Maximum number of results
            
        Returns:
            List of similar Block instances for this agent
        """
        # This would use Neo4j's vector similarity search
        # filtered by agent_id
        query = """
        MATCH (b:Block)
        WHERE b.agent_id = $agent_id
        WITH b, gds.similarity.cosine(b.embedding_v1, $embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN b
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(
            query,
            {
                'agent_id': self.agent_id,
                'embedding': embedding,
                'threshold': threshold,
                'limit': limit
            }
        )
        
        # Convert results to Block instances
        blocks = []
        for row in results:
            block_data = row[0]
            block = Block.inflate(block_data)
            blocks.append(block)
        
        return blocks
