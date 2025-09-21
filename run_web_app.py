#!/usr/bin/env python3
"""
Launch script for the LLM Chat Web Interface

Usage:
    python run_web_app.py [--port PORT] [--host HOST]

Default: http://localhost:8000
"""

import sys
import os
import argparse
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Launch the LLM Chat Web Interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print("Please create a .env file with your ANTHROPIC_API_KEY")
        print("Example: ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env file!")
        print("Please add your API key to the .env file")
        sys.exit(1)

    data_dir = Path(__file__).parent / "data" / "sessions"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting LLM Chat Interface...")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop the server\n")

    if not args.no_browser:
        import threading
        import time

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "src.web_app:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()