# toctalk

A minimal LLM chat framework for research on task-oriented conversation. Built with Python/FastAPI backend and vanilla JS frontend.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Usage

```bash
# Start the web server
source venv/bin/activate
python run_web_app.py
```

Open http://localhost:8000 in your browser to start chatting.

## Project Structure

```
toctalk/
├── src/                    # Core Python modules
│   ├── llm_client.py      # Claude API wrapper
│   ├── chat_session.py    # Session data models
│   ├── session_storage.py # JSON persistence
│   ├── chat_manager.py    # Chat orchestration
│   └── web_server.py      # FastAPI server
├── data/                   # Runtime data
│   └── sessions/          # Saved chat sessions (JSON)
├── static/                 # Web interface
│   ├── index.html         # Single-page app
│   ├── style.css          # UI styling
│   └── app.js             # Frontend logic
├── tmp_scripts/           # Development utilities
└── planning/              # Project documentation
```

## Features

- Real-time streaming chat with Claude models
- Automatic session persistence
- Session management (create, load, delete)
- Server-Sent Events for streaming responses
- No frontend framework dependencies

## Development

Sessions are automatically saved to `data/sessions/` as JSON files. Each session includes metadata (tags, description) and full conversation history with timestamps.

Default model: `claude-3-5-haiku-20241022` (configurable in `src/llm_client.py`)