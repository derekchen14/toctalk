import os
from typing import List, Dict, Optional, Generator, AsyncGenerator
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
import asyncio

load_dotenv()


class ClaudeLLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or provided")

        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], stream: bool = False,
             temperature: float = 0.7, max_tokens: int = 4096) -> Generator[str, None, None] | str:

        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        if stream:
            response = self.client.messages.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in response:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
        else:
            response = self.client.messages.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text

    async def chat_async(self, messages: List[Dict[str, str]], stream: bool = False,
                        temperature: float = 0.7, max_tokens: int = 4096) -> str:
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = await self.async_client.messages.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text

    async def chat_stream_async(self, messages: List[Dict[str, str]],
                                temperature: float = 0.7, max_tokens: int = 4096) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        stream = await self.async_client.messages.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

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