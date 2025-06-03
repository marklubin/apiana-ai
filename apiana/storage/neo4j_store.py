from datetime import datetime
from typing import List

from neomodel import config as neomodel_config
from neomodel import db

from apiana.types.common import Conversation
from apiana.types.configuration import Neo4jConfig
from apiana.types.memory_block import Block, Tag, Grounding


class Neo4jMemoryStore:
    def __init__(self, config: Neo4jConfig):
        config_url = (
            f"bolt://{config.username}:{config.password}@{config.host}:{config.port}"
        )
        neomodel_config.DATABASE_URL = config_url
        db.install_all_labels()

    def store_convo(
        self,
        conversation: Conversation,
        summary: str,
        embeddings: List[float],
        tags: List[str],
    ) -> Block:
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
                "external_id": conversation.openai_conversation_id,
                "external_label": conversation.title,
                "external_source": "conversation",
            }
        )[0]

        # Create a parent Block for the conversation
        block = Block(
            content=summary,
            created_at=now,
            updated_at=now,
            embedding_v1=embeddings,
            tags=db_tags,
            block_type="experience",
            experience_type="conversation",
        ).save()

        block.grounded_by.connect(grounding)
        [block.tagged_with.connect(tag) for tag in db_tags.values()]
        return block
