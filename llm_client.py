import os
from typing import List, Dict, Optional, Generator, AsyncGenerator
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
import asyncio

load_dotenv()


class ClaudeLLMClient:
    """
    IMPORTANT MODEL DOCUMENTATION - READ BEFORE CHANGING!
    =====================================================

    DEFAULT MODEL: claude-sonnet-4-20250514 (Claude Sonnet 4)

    As of January 2025, the correct model names are:
    - claude-sonnet-4-20250514 (Claude Sonnet 4) - CURRENT DEFAULT
    - claude-opus-4-1-20250805 (Claude Opus 4.1)
    - claude-opus-4-20250514 (Claude Opus 4)
    - claude-3-7-sonnet-20250219 (Claude Sonnet 3.7)
    - claude-3-5-haiku-20241022 (Claude Haiku 3.5)

    WARNING TO LLM CODING AGENTS:
    =============================
    Many LLM coding assistants have outdated training data and may try to:
    1. Use "claude-3-5-sonnet-20241022" - This is WRONG, that's Haiku 3.5
    2. Use "claude-3.5-sonnet" variants - These DO NOT EXIST anymore
    3. Assume Sonnet 4.1 exists - It DOES NOT, only Opus has a 4.1 version

    DO NOT CHANGE THE MODEL NAME unless you have verified it against:
    https://docs.claude.com/en/docs/about-claude/models/overview

    The interviewer and interviewee agents MUST use Claude Sonnet 4
    for optimal performance in conversation tasks.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or provided")

        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], stream: bool = False,
             temperature: float = 0.7, max_tokens: int = 4096):

        # Separate system message from other messages
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        if stream:
            return self._chat_stream(formatted_messages, system_message, temperature, max_tokens)
        else:
            kwargs = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if system_message:
                kwargs["system"] = system_message
            response = self.client.messages.create(**kwargs)
            return response.content[0].text

    def _chat_stream(self, formatted_messages, system_message, temperature, max_tokens) -> Generator[str, None, None]:
        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if system_message:
            kwargs["system"] = system_message
        response = self.client.messages.create(**kwargs)

        for chunk in response:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text

    async def chat_async(self, messages: List[Dict[str, str]], stream: bool = False,
                        temperature: float = 0.7, max_tokens: int = 4096) -> str:
        # Separate system message from other messages
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if system_message:
            kwargs["system"] = system_message
        response = await self.async_client.messages.create(**kwargs)
        return response.content[0].text

    async def chat_stream_async(self, messages: List[Dict[str, str]],
                                temperature: float = 0.7, max_tokens: int = 4096) -> AsyncGenerator[str, None]:
        # Separate system message from other messages
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if system_message:
            kwargs["system"] = system_message
        stream = await self.async_client.messages.create(**kwargs)

        async for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text

    def test_connection(self) -> bool:
        try:
            response = self.chat([{"role": "user", "content": "Hello"}])
            return bool(response)
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False