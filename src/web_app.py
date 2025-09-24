from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
from typing import Optional, List, AsyncGenerator
from pathlib import Path
import uuid
from datetime import datetime

from .chat_manager import ChatManager
from .chat_session import ChatSession
from .quirkbot import load_personas, get_random_persona, generate_system_prompt, extract_persona_name_age, generate_interviewer_system_prompt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

chat_manager = ChatManager()

class NewSessionRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class MessageRequest(BaseModel):
    session_id: str
    message: str

class SessionInfo(BaseModel):
    session_id: str
    description: Optional[str]
    tags: List[str]
    created_at: str
    updated_at: str
    message_count: int

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Chat Interface</h1><p>Static files not found</p>"

@app.post("/api/sessions/new")
async def create_new_session(request: NewSessionRequest):
    session_id = str(uuid.uuid4())
    session = ChatSession(
        session_id=session_id,
        description=request.description,
        tags=request.tags or []
    )
    chat_manager.storage.save_session(session)
    return {
        "session_id": session_id,
        "created_at": session.created_at.isoformat()
    }

@app.post("/api/quirkbot/random")
async def create_quirkbot_session():
    # Load personas from benchmark
    personas_path = Path(__file__).parent.parent / "benchmarks" / "quirkbot_benchmark_v0.json"
    if not personas_path.exists():
        raise HTTPException(status_code=404, detail="Persona benchmark file not found")

    # Get a random persona
    personas_data = load_personas(str(personas_path))
    persona = get_random_persona(personas_data)

    # Extract name and age for display
    name, age = extract_persona_name_age(persona['natural_narrative'])

    # Generate system prompt
    system_prompt = generate_system_prompt(persona)

    # Create a new session with the persona
    session_id = str(uuid.uuid4())
    description = f"Quirkbot: {name}, {age}"

    session = ChatSession(
        session_id=session_id,
        description=description,
        tags=["quirkbot"]
    )

    # Store persona info in session metadata
    session.metadata = {
        "is_quirkbot": True,
        "persona_id": persona["bio_id"],
        "persona_name": name,
        "persona_age": age,
        "system_prompt": system_prompt
    }

    chat_manager.storage.save_session(session)

    return {
        "session_id": session_id,
        "persona_name": name,
        "persona_age": age,
        "persona_id": persona["bio_id"],
        "created_at": session.created_at.isoformat()
    }

@app.post("/api/quirkbot/interviewer")
async def create_interviewer_session():
    # Generate interviewer system prompt
    system_prompt = generate_interviewer_system_prompt()

    # Create a new session for the interviewer
    session_id = str(uuid.uuid4())
    description = "Quirkbot Interviewer"

    session = ChatSession(
        session_id=session_id,
        description=description,
        tags=["quirkbot-interviewer"]
    )

    # Store interviewer info in session metadata
    session.metadata = {
        "is_interviewer": True,
        "system_prompt": system_prompt
    }

    chat_manager.storage.save_session(session)

    return {
        "session_id": session_id,
        "description": description,
        "created_at": session.created_at.isoformat()
    }

@app.get("/api/sessions")
async def list_sessions():
    sessions = chat_manager.storage.list_sessions()
    result = []
    for session_data in sessions:
        result.append(SessionInfo(
            session_id=session_data["session_id"],
            description=session_data.get("description"),
            tags=session_data.get("tags", []),
            created_at=session_data["created_at"],
            updated_at=session_data["updated_at"],
            message_count=session_data.get("message_count", 0)
        ))
    return result

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = chat_manager.storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()

@app.post("/api/sessions/{session_id}/messages")
async def send_message(session_id: str, request: MessageRequest):
    session = chat_manager.storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_manager.current_session = session
    session.add_message("user", request.message)

    # Prepare messages with system prompt if quirkbot or interviewer
    messages = [msg.model_dump() for msg in session.messages]
    if hasattr(session, 'metadata') and session.metadata:
        if session.metadata.get('is_quirkbot') or session.metadata.get('is_interviewer'):
            system_prompt = session.metadata.get('system_prompt')
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

    response = await chat_manager.llm_client.chat_async(
        messages=messages,
        stream=False
    )

    session.add_message("assistant", response)
    chat_manager.storage.save_session(session)

    return {"message": response}

async def generate_stream(session_id: str, message: str) -> AsyncGenerator[str, None]:
    try:
        session = chat_manager.storage.load_session(session_id)
        if not session:
            yield json.dumps({'error': 'Session not found'})
            return

        chat_manager.current_session = session
        session.add_message("user", message)

        # Prepare messages with system prompt if quirkbot
        messages = [msg.model_dump() for msg in session.messages]
        if hasattr(session, 'metadata') and session.metadata and session.metadata.get('is_quirkbot'):
            system_prompt = session.metadata.get('system_prompt')
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

        full_response = ""
        async for chunk in chat_manager.llm_client.chat_stream_async(
            messages=messages
        ):
            full_response += chunk
            yield json.dumps({'chunk': chunk})

        session.add_message("assistant", full_response)
        chat_manager.storage.save_session(session)

        yield json.dumps({'done': True, 'full_response': full_response})
    except Exception as e:
        import traceback
        error_msg = f"Error during streaming: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        yield json.dumps({'error': error_msg})

@app.post("/api/sessions/{session_id}/messages/stream")
async def send_message_stream(session_id: str, request: MessageRequest):
    return EventSourceResponse(generate_stream(session_id, request.message))

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_path = Path(f"data/sessions/{session_id}.json")
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    session_path.unlink()
    return {"message": "Session deleted successfully"}