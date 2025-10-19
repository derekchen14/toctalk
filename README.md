# toctalk

A minimal LLM chat framework for research on task-oriented conversation. Built with Python/FastAPI backend and vanilla JS frontend.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install flash_attn --no-build-isolation

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Usage

### Interactive Web Chat

```bash
# Start the web server
source .venv/bin/activate
python run_web_app.py
```
Open http://localhost:8000 in your browser to start chatting.

### Model Training

```bash
# Train the first time model
python training.py --allow_download --output_dir ./results --dataset_path your_dataset
# All future calls, especially for running multiple trials as once
python training.py --use_checkpoint
```

#### Tensorboard Monitoring

Tensorboard logging is automatically enabled when `--output_dir` is specified:

```bash
# Start training (Tensorboard auto-enabled)
python training.py --method grpo --output_dir ./results --dataset_path your_dataset --model_size small

# View training metrics (separate terminal)
tensorboard --logdir ./results
# Open http://localhost:6006
```

**Experiment Naming**: Runs are organized as `{model_name}_{model_size}_LR{learning_rate}_v{seed}`

**Available Metrics**:
- Training loss, learning rate, gradient norms
- Custom reward metrics (format compliance, accuracy)
- Training steps per second

**Cloud Usage (Lambda Labs)**:
```bash
# Create SSH tunnel from local machine
ssh -L 6006:localhost:6006 username@lambda-instance-ip

# Start Tensorboard on Lambda instance
tensorboard --logdir ./results --bind_all
# Access via http://localhost:6006 on local machine
```

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
- Quirkbot personas for conversational experiments
- Quirkbot interviewer with note-taking capabilities
- No frontend framework dependencies

## Quirkbot Interviewer

The interviewer agent helps discover quirky biographical facts through conversation:

```python
# Create an interviewer session
POST /api/quirkbot/interviewer

# The interviewer takes notes using <note> tags during conversation
# Notes can be parsed using src/note_parser.py utilities
```

See `examples/interviewer_usage.py` for a complete example.

## Quirkbot Evaluation

Evaluate how well an interviewer discovered embedded facts in a QuirkBot session:

```bash
# Evaluate a single session
python src/quirkbot_evaluator.py data/sessions/SESSION_ID.json

# Generate a test interview session with automated agents
python tmp_scripts/run_interviewer_session.py

# The evaluator outputs:
# - Discovery rate (% of facts found)
# - Facts per turn
# - Facts per 1k tokens
# - Detailed analysis of each fact
# - JSON results saved to data/evaluations/
```

The evaluator uses LLM-based semantic matching to determine if facts were discovered, even if worded differently in the interviewer's notes.

## Development

Sessions are automatically saved to `data/sessions/` as JSON files. Each session includes metadata (tags, description) and full conversation history with timestamps.

Default model: `claude-3-5-haiku-20241022` (configurable in `src/llm_client.py`)

## Possible Datasets

These datasets are directly available on Huggingface and conversational in nature:
 - goendalf666/sales-conversations (3,412 rows)
     - train the model to sound more like a salesperson
 - flammenai/casual-conversation-DPO (3,725 rows)
     - train the model to sound more casual, or more like AI slop
 - DeepMostInnovations/saas-sales-conversations (100K rows)
     - sounds like a salesperson, but also includes:
     - Engagement metrics, sales effectiveness scores, Conversion outcome
 
