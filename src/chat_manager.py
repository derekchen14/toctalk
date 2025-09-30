from typing import Optional, Generator
from .llm_client import ClaudeLLMClient
from .chat_session import ChatSession
from .session_storage import SessionStorage


class ChatManager:
    def __init__(self, storage: Optional[SessionStorage] = None, llm_client: Optional[ClaudeLLMClient] = None):
        self.storage = storage or SessionStorage()
        # NOTE: Default LLM client uses claude-sonnet-4-20250514 (Claude Sonnet 4)
        # DO NOT change to "claude-3-5-sonnet" variants - they don't exist!
        # See llm_client.py for full model documentation
        self.llm_client = llm_client or ClaudeLLMClient()
        self.current_session: Optional[ChatSession] = None

    def create_session(self, description: Optional[str] = None, tags: Optional[list] = None, system_prompt: Optional[str] = None) -> ChatSession:
        self.current_session = ChatSession(
            description=description,
            tags=tags or [],
            model_used=self.llm_client.model
        )
        # Add system message if provided
        if system_prompt:
            self.current_session.add_message("system", system_prompt)
        return self.current_session

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        session = self.storage.load_session(session_id)
        if session:
            self.current_session = session
        return session

    def save_current_session(self) -> Optional[str]:
        if self.current_session:
            return self.storage.save_session(self.current_session)
        return None

    def send_message(self, content: str, stream: bool = False, system_prompt: Optional[str] = None):
        if not self.current_session:
            self.create_session(system_prompt=system_prompt)

        self.current_session.add_message("user", content)

        messages = self.current_session.get_messages_for_api()

        if stream:
            return self._send_message_stream(messages)
        else:
            response = self.llm_client.chat(messages, stream=False)
            self.current_session.add_message("assistant", response)
            self.save_current_session()
            return response

    def _send_message_stream(self, messages) -> Generator[str, None, None]:
        full_response = ""
        for chunk in self.llm_client.chat(messages, stream=True):
            full_response += chunk
            yield chunk

        self.current_session.add_message("assistant", full_response)
        self.save_current_session()

    def clear_current_session(self):
        if self.current_session:
            self.current_session.clear_messages()

    def list_sessions(self):
        return self.storage.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        return self.storage.delete_session(session_id)