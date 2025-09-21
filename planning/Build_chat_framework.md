# Build Simple LLM Chat Framework

## Instructions for Using This Document

1. Work in sequence: Complete phases in order (Phase 0 → Phase 1 → ...)
2. Mark progress: Use `- [x]` to mark completed subtasks
3. Fill Developer Notes: Add notes at beginning and end of each phase as work progresses
4. Sanity checks: Complete ALL sanity checks before moving to next phase
5. Ask before deviating: If unexpected issues arise, ask permission to deviate from plan
6. Use `tmp_scripts/`: Create any temporary scripts/tests/configs here to keep the repo tidy

---

## Context & Goal

Build a minimal LLM chat framework for research on task-oriented conversation. The system needs to support generating, conducting, and saving chat sessions with metadata for evaluation and training purposes. A simple local web interface will enable human-AI interaction. Focus on Claude models initially via the Anthropic API.

---

## Phase 0: Core Chat Backend

**Goal**: Implement the foundational Python backend for LLM chat interactions with session management and data persistence.

### Developer Notes - Phase 0 Start

**Important Environment Setup:**
- Always create and use a virtual environment: `python3 -m venv venv`
- Activate venv before any work: `source venv/bin/activate`
- Install dependencies in venv: `pip install -r requirements.txt`
- Create `.env` file with `ANTHROPIC_API_KEY=your-key-here`
- All testing must be done within the activated venv

### Tasks

- [x] Create project structure (`src/`, `data/`, `tmp_scripts/`)
- [x] Set up Python environment with requirements.txt (anthropic, python-dotenv, pydantic)
- [x] Implement LLM client wrapper class for Claude API
- [x] Create Chat Session data model with metadata support
- [x] Implement session storage system (JSON files in `data/sessions/`)
- [x] Build basic chat loop functionality
- [x] Create simple CLI test script in `tmp_scripts/`

### Sanity Checks - Phase 0

- [x] Can successfully load API keys from .env
- [x] Can make a test call to Claude API
- [x] Can create, save, and load a chat session from disk
- [x] CLI test script completes a multi-turn conversation

### Developer Notes - Phase 0 End

**Completed Implementation:**
- Core backend structure created with modular Python components
- Virtual environment setup with dependencies (anthropic 0.39.0, pydantic 2.9.2, python-dotenv 1.0.1)
- Implemented ClaudeLLMClient with streaming support
- Created Pydantic-based ChatSession model with metadata and timestamps
- Built JSON-based SessionStorage for persistent chat history
- Developed ChatManager to orchestrate chat operations
- CLI test script ready in `tmp_scripts/test_cli.py`

**Next Steps:**
- User needs to add ANTHROPIC_API_KEY to .env file
- Run sanity checks: `source venv/bin/activate && python tmp_scripts/test_cli.py`
- Phase 1 can begin once sanity checks pass

---

## Phase 1: Web Interface

**Goal**: Create a simple local web interface for human-AI chat interactions.

### Developer Notes - Phase 1 Start

**Phase 1 Implementation Started**
- Building on top of Phase 0's backend components
- Using FastAPI for async web server with SSE support
- Creating single-page application with vanilla JS

### Tasks

- [x] Set up Flask/FastAPI backend server
- [x] Create API endpoints for chat operations (new session, send message, list sessions)
- [x] Build minimal HTML/CSS/JS frontend (single page, no framework)
- [x] Implement real-time message streaming (SSE or WebSocket)
- [x] Add session management UI (new, load, save)
- [x] Create launch script for starting the web app

### Sanity Checks - Phase 1

- [x] Web server starts on localhost without errors
- [x] Can create new chat session from UI
- [x] Messages stream in real-time as AI responds
- [x] Session history persists after browser refresh
- [x] Can load and continue previous sessions

### Developer Notes - Phase 1 End

**Phase 1 Implementation Completed Successfully**

**Core Implementation:**
- FastAPI server implemented with full async support
- API endpoints created for all chat operations:
  - POST /api/sessions/new - Create new session
  - GET /api/sessions - List all sessions
  - GET /api/sessions/{id} - Get session details
  - POST /api/sessions/{id}/messages/stream - Send message with SSE streaming
  - DELETE /api/sessions/{id} - Delete session
- Single-page HTML/CSS/JS interface built with:
  - Real-time message streaming using Server-Sent Events
  - Session management sidebar with auto-refresh
  - Modal for new session creation with tags/description
  - Auto-scrolling chat interface with typing indicators
- Launch script created: `python run_web_app.py`
- Updated dependencies: FastAPI, uvicorn, sse-starlette, upgraded anthropic SDK

**Key Issues Resolved During Implementation:**
1. **Anthropic SDK Version**: Upgraded from 0.39.0 to latest version to fix compatibility issues
2. **SSE Double-Prefixing**: EventSourceResponse automatically adds "data: " prefix - removed manual prefixing in generator
3. **CSS Layout Issues**: Fixed message bubble alignment with proper flexbox justification and max-width on wrapper div
4. **Async Streaming**: Added AsyncAnthropic client and async methods to LLMClient for proper streaming support
5. **Session State Management**: Fixed ChatManager methods - use session.add_message() not chat_manager.add_message()
6. **Default Model**: Changed from claude-3-5-sonnet to claude-3-5-haiku-20241022 for faster/cheaper responses
7. **Pydantic Warning**: Added ConfigDict(protected_namespaces=()) to ChatSession model to resolve field name conflict

**Architecture Insights for Next Phase:**
- Sessions auto-save after each message exchange - no manual save needed
- SessionStorage uses JSON files in data/sessions/ directory
- Frontend uses vanilla JS with fetch API for simplicity
- SSE streaming provides good UX for real-time responses
- Error handling added at both server and client levels

**Testing Utilities Created:**
- `tmp_scripts/test_cli.py` - CLI testing from Phase 0
- `tmp_scripts/test_streaming.py` - SSE streaming verification

**To run the application:**
```bash
source venv/bin/activate
python run_web_app.py
```
Web interface opens at http://localhost:8000

**Post-Phase 1 Improvements (Session Auto-Save Enhancement):**
- **Fixed Sessions Panel Issue**: Sessions now correctly appear in the Sessions panel after being saved
  - Root cause: `/api/sessions` endpoint was incorrectly iterating over session IDs when `list_sessions()` already returned full session data
  - Fix: Updated endpoint to directly use the session dictionaries returned by `list_sessions()`
- **Simplified User Experience**:
  - Removed manual "Save" button - sessions auto-save after every message
  - Removed "Refresh" button - sessions list updates automatically
  - Sessions are created automatically on first message - no need to manually create session first
  - Message input enabled by default - users can start typing immediately
  - Session appears in panel immediately after first message is sent
- **Current Flow**:
  1. User opens app and starts typing (no setup required)
  2. First message automatically creates a new session
  3. Session appears in Sessions panel immediately
  4. All subsequent messages auto-save
  5. Users can click any session in panel to load full chat history

---