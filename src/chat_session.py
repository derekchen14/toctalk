from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSession(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    model_used: Optional[str] = None

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def to_dict(self) -> dict:
        data = self.model_dump()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        for msg in data['messages']:
            msg['timestamp'] = msg['timestamp'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        for msg in data['messages']:
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
        return cls(**data)

    def clear_messages(self):
        self.messages = []
        self.updated_at = datetime.now()