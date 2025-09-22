import json
import random
from typing import Dict, List, Optional, Any
from pathlib import Path


def load_personas(filepath: str) -> Dict[str, Any]:
    """Load personas from the benchmark JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_random_persona(personas_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get a random persona from the loaded data."""
    return random.choice(personas_data['biographies'])


def get_persona_by_id(personas_data: Dict[str, Any], bio_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific persona by bio_id."""
    for bio in personas_data['biographies']:
        if bio['bio_id'] == bio_id:
            return bio
    return None


def generate_system_prompt(persona: Dict[str, Any]) -> str:
    """Generate a system prompt for the quirkbot agent based on a persona."""

    # Extract persona information
    narrative = persona['natural_narrative']
    facts = persona['embedded_facts']

    # Create the system prompt
    system_prompt = f"""You are participating in a task-oriented conversation experiment designed to measure conversational skills. You will play the role of a synthetic person based on the following biographical information.

BIOGRAPHY:
{narrative}

IMPORTANT EMBEDDED FACTS:
These are quirky facts about you that should emerge naturally in conversation when the topic arises:
"""

    for fact in facts:
        system_prompt += f"- {fact['fact_text']} (Category: {fact['category']})\n"

    system_prompt += """
INSTRUCTIONS:
1. You are aware you're part of an experiment to measure task-oriented conversational skills
2. Play the part! Embody this persona naturally and authentically
3. Let the interviewer interview you. Don't ask the interviewer questions about them.
4. When relevant topics come up in conversation, naturally reveal the embedded facts
5. Don't force the facts into conversation - let them emerge organically
6. Respond as this person would, with their personality, background, and experiences
7. Most importantly: Be conversational! Don't volunteer biographical facts unless it comes up naturally in the conversation.

Remember: You ARE this person for the duration of our conversation."""

    return system_prompt


def extract_persona_name_age(narrative: str) -> tuple[str, str]:
    """Extract name and age from the narrative text."""
    lines = narrative.strip().split('\n')
    if lines:
        # First line typically contains "Name, Age"
        first_line = lines[0]
        if ',' in first_line:
            parts = first_line.rsplit(',', 1)
            name = parts[0].strip()
            age = parts[1].strip() if len(parts) > 1 else "Unknown"
            return name, age
    return "Unknown", "Unknown"