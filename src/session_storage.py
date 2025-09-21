import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .chat_session import ChatSession


class SessionStorage:
    def __init__(self, storage_path: str = "data/sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        return self.storage_path / f"{session_id}.json"

    def save_session(self, session: ChatSession) -> str:
        session_path = self._get_session_path(session.session_id)
        session_data = session.to_dict()

        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return str(session_path)

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return None

        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            return ChatSession.from_dict(session_data)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def list_sessions(self) -> List[dict]:
        sessions = []
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                sessions.append({
                    "session_id": session_data.get("session_id"),
                    "created_at": session_data.get("created_at"),
                    "updated_at": session_data.get("updated_at"),
                    "description": session_data.get("description", ""),
                    "tags": session_data.get("tags", []),
                    "message_count": len(session_data.get("messages", []))
                })
            except Exception as e:
                print(f"Error reading session file {session_file}: {e}")
                continue

        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        session_path = self._get_session_path(session_id)

        if session_path.exists():
            try:
                session_path.unlink()
                return True
            except Exception as e:
                print(f"Error deleting session {session_id}: {e}")
                return False

        return False

    def session_exists(self, session_id: str) -> bool:
        return self._get_session_path(session_id).exists()