#!/usr/bin/env python3
import sys
import os
import webbrowser
import uvicorn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.arguments import parse_arguments

def main():
    args = parse_arguments()

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
    print(f"Press Ctrl+C to stop the server\n")    
    uvicorn.run("src.web_app:app", port=args.port, reload=True, log_level="info")

if __name__ == "__main__":
    main()