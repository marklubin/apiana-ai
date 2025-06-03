"""
common.py  – data models shared across the voice‑assistant‑tui project.

Key rules
---------
1. **No Optional[...] annotations** – every field is required *but* each
   has a sensible default so callers can instantiate the model without
   arguments if desired.
2. Lean on Pydantic V2’s built‑in parsing instead of custom validators.
3. Provide thin helper `to_dict()` / `to_json()` wrappers that simply
   forward to `model_dump` / `model_dump_json` for convenience.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#                               Message Model                                 #
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    id: str = ""
    role: str = "user"  # e.g. user / assistant / system
    type: str = "text"  # text / image / etc.
    content: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provider_model_id: str = ""
    provider_request_id: str = ""

    # convenience helpers --------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """JSON‑serialisable dict."""
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent)


# --------------------------------------------------------------------------- #
#                             Conversation Model                              #
# --------------------------------------------------------------------------- #
class Conversation(BaseModel):
    title: str = "untitled conversation"
    messages: List[Message] = Field(default_factory=list)
    openai_conversation_id: str = ""
    create_time: datetime = Field(default_factory=datetime.utcnow)
    update_time: datetime = Field(default_factory=datetime.utcnow)

    # convenience helpers --------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent)
