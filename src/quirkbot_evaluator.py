"""
QuirkBot Evaluation Framework
Evaluates how well an interviewer discovered embedded facts in a conversation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from note_parser import parse_notes_from_transcript
from llm_client import ClaudeLLMClient
import tiktoken


def extract_embedded_facts(session_data: Dict) -> List[Dict]:
    """
    Extract embedded facts from a QuirkBot session.

    Args:
        session_data: The loaded session JSON data

    Returns:
        List of fact dictionaries with 'fact_text' and 'category'
    """
    embedded_facts = []

    # First priority: Check metadata for persona_data with embedded_facts
    if 'metadata' in session_data and 'persona_data' in session_data['metadata']:
        persona_data = session_data['metadata']['persona_data']
        if 'embedded_facts' in persona_data:
            return persona_data['embedded_facts']

    # Check metadata for embedded_facts directly (interviewer sessions)
    if 'metadata' in session_data and 'embedded_facts' in session_data['metadata']:
        return session_data['metadata']['embedded_facts']

    # Check metadata for system_prompt (fallback for older sessions)
    if 'metadata' in session_data and 'system_prompt' in session_data['metadata']:
        system_prompt = session_data['metadata']['system_prompt']
        facts = parse_facts_from_prompt(system_prompt)
        if facts:
            return facts

    # Check metadata for interviewee_system_prompt (interviewer sessions)
    if 'metadata' in session_data and 'interviewee_system_prompt' in session_data['metadata']:
        system_prompt = session_data['metadata']['interviewee_system_prompt']
        facts = parse_facts_from_prompt(system_prompt)
        if facts:
            return facts

    # Check for system message in messages
    messages = session_data.get('messages', [])
    for message in messages:
        if message.get('role') == 'system':
            facts = parse_facts_from_prompt(message.get('content', ''))
            if facts:
                return facts

    return embedded_facts


def parse_facts_from_prompt(prompt_text: str) -> List[Dict]:
    """
    Parse embedded facts from a system prompt text.

    Args:
        prompt_text: The system prompt containing embedded facts

    Returns:
        List of fact dictionaries
    """
    facts = []

    # Find the embedded facts section - support multiple markers
    facts_section = None
    if 'IMPORTANT EMBEDDED FACTS:' in prompt_text:
        facts_section = prompt_text.split('IMPORTANT EMBEDDED FACTS:')[1]
    elif '## EMBEDDED FACTS' in prompt_text:
        facts_section = prompt_text.split('## EMBEDDED FACTS')[1]

    if facts_section:
        # Split at next section marker to get just the facts part
        for marker in ['INSTRUCTIONS:', '## IMPROVISATION GUIDELINES', '## REMEMBER', '## YOUR ROLE', '## CRITICAL']:
            if marker in facts_section:
                facts_section = facts_section.split(marker)[0]
                break

        # Parse individual facts (they start with -)
        fact_lines = [line.strip() for line in facts_section.split('\n') if line.strip().startswith('-')]

        for line in fact_lines:
            # Remove leading dash
            line = line[1:].strip()

            # Extract fact text and category using regex
            # Pattern: "fact text (Category: category_name)"
            match = re.match(r'(.*?)\s*\(Category:\s*(.*?)\)', line)
            if match:
                facts.append({
                    'fact_text': match.group(1).strip(),
                    'category': match.group(2).strip()
                })
            else:
                # Fallback: treat whole line as fact text
                facts.append({
                    'fact_text': line,
                    'category': 'Unknown'
                })

    return facts


def evaluate_fact_discovery(fact: Dict, notes: List[str], llm_client: ClaudeLLMClient) -> Tuple[bool, str]:
    """
    Use LLM to evaluate if a specific fact was discovered based on the notes.

    Args:
        fact: The embedded fact to check
        notes: List of interviewer notes
        llm_client: LLM client for evaluation

    Returns:
        Tuple of (discovered: bool, reasoning: str)
    """
    if not notes:
        return False, "No notes found"

    # Combine all notes into a single text
    notes_text = "\n".join(notes)

    prompt = f"""You are evaluating whether an interviewer discovered a specific fact about a person during a conversation.

EMBEDDED FACT TO CHECK:
{fact['fact_text']} (Category: {fact['category']})

INTERVIEWER'S NOTES:
{notes_text}

TASK: Determine if the interviewer discovered this specific fact based on their notes.

Consider a fact "discovered" if:
- The notes explicitly mention the fact or very similar information
- The notes describe the same event/skill/experience even if worded differently
- The core information matches even if details differ slightly

Consider a fact "NOT discovered" if:
- The notes don't mention this fact at all
- The notes only mention vaguely related topics without the specific fact
- The information is too different to be considered the same fact

Respond in the following JSON format:
{{
    "discovered": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of your decision"
}}"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.chat(messages, stream=False, temperature=0.3)
        result = json.loads(response)
        return result.get('discovered', False), result.get('reasoning', 'No reasoning provided')
    except (json.JSONDecodeError, Exception) as e:
        # Fallback to simple keyword matching if LLM fails
        fact_words = set(fact['fact_text'].lower().split())
        notes_words = set(notes_text.lower().split())
        overlap = len(fact_words & notes_words) / len(fact_words) if fact_words else 0
        discovered = overlap > 0.5
        return discovered, f"Fallback evaluation (word overlap: {overlap:.1%})"


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens in a text string using tiktoken.

    Args:
        text: Text to count tokens for
        model: Encoding model to use (default is cl100k_base for Claude/GPT-4)

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to word-based approximation (roughly 1.3 tokens per word)
        words = len(text.split())
        return int(words * 1.3)


def evaluate_session(session_path: str, verbose: bool = True) -> Dict:
    """
    Evaluate a QuirkBot session to measure fact discovery.

    Args:
        session_path: Path to the session JSON file
        verbose: Whether to print detailed output

    Returns:
        Dictionary with evaluation metrics
    """
    # Load session data
    with open(session_path, 'r') as f:
        session_data = json.load(f)

    # Extract embedded facts
    embedded_facts = extract_embedded_facts(session_data)
    if not embedded_facts and verbose:
        print("Warning: No embedded facts found in session")

    # Parse notes from transcript
    messages = session_data.get('messages', [])
    all_notes = parse_notes_from_transcript(messages)

    # Count conversation turns (excluding system messages)
    turns = sum(1 for m in messages if m.get('role') in ['user', 'assistant'])

    # Count tokens in conversation
    total_tokens = 0
    interviewer_tokens = 0
    interviewee_tokens = 0

    for message in messages:
        role = message.get('role')
        content = message.get('content', '')
        tokens = count_tokens(content)

        if role in ['user', 'assistant']:
            total_tokens += tokens

            # Track interviewer vs interviewee tokens if metadata available
            metadata = message.get('metadata', {})
            if metadata.get('is_interviewer'):
                interviewer_tokens += tokens
            elif metadata.get('is_interviewee') or role == 'assistant':
                interviewee_tokens += tokens
            elif role == 'user':
                interviewer_tokens += tokens

    # Initialize LLM client for evaluation
    llm_client = ClaudeLLMClient()

    # Evaluate each embedded fact
    results = {
        'session_id': session_data.get('session_id', 'unknown'),
        'total_facts': len(embedded_facts),
        'facts_discovered': 0,
        'discovery_details': [],
        'total_turns': turns,
        'total_notes': len(all_notes),
        'total_tokens': total_tokens,
        'interviewer_tokens': interviewer_tokens,
        'interviewee_tokens': interviewee_tokens
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"QuirkBot Session Evaluation")
        print(f"{'='*60}")
        print(f"Session ID: {results['session_id']}")
        print(f"Total embedded facts: {results['total_facts']}")
        print(f"Total conversation turns: {results['total_turns']}")
        print(f"Total tokens: {results['total_tokens']:,}")
        print(f"  - Interviewer tokens: {results['interviewer_tokens']:,}")
        print(f"  - Interviewee tokens: {results['interviewee_tokens']:,}")
        print(f"Total notes taken: {results['total_notes']}")
        print(f"\n{'='*60}")
        print("Fact Discovery Analysis")
        print(f"{'='*60}")

    # Get all note contents for evaluation
    note_contents = [note['note'] for note in all_notes]

    for i, fact in enumerate(embedded_facts, 1):
        discovered, reasoning = evaluate_fact_discovery(fact, note_contents, llm_client)

        results['discovery_details'].append({
            'fact': fact['fact_text'],
            'category': fact['category'],
            'discovered': discovered,
            'reasoning': reasoning
        })

        if discovered:
            results['facts_discovered'] += 1

        if verbose:
            status = "✓ DISCOVERED" if discovered else "✗ NOT FOUND"
            print(f"\nFact #{i}: {fact['fact_text']}")
            print(f"Category: {fact['category']}")
            print(f"Status: {status}")
            print(f"Reasoning: {reasoning}")

    # Calculate metrics
    if results['total_facts'] > 0:
        results['discovery_rate'] = results['facts_discovered'] / results['total_facts']
    else:
        results['discovery_rate'] = 0.0

    if results['total_turns'] > 0:
        results['facts_per_turn'] = results['facts_discovered'] / results['total_turns']
    else:
        results['facts_per_turn'] = 0.0

    if results['total_tokens'] > 0:
        results['facts_per_token'] = results['facts_discovered'] / results['total_tokens']
        results['facts_per_1k_tokens'] = (results['facts_discovered'] / results['total_tokens']) * 1000
    else:
        results['facts_per_token'] = 0.0
        results['facts_per_1k_tokens'] = 0.0

    # Calculate interviewer efficiency (facts per interviewer token)
    if results['interviewer_tokens'] > 0:
        results['interviewer_efficiency'] = results['facts_discovered'] / results['interviewer_tokens']
        results['interviewer_facts_per_1k_tokens'] = (results['facts_discovered'] / results['interviewer_tokens']) * 1000
    else:
        results['interviewer_efficiency'] = 0.0
        results['interviewer_facts_per_1k_tokens'] = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print("Summary Metrics")
        print(f"{'='*60}")
        print(f"Facts discovered: {results['facts_discovered']}/{results['total_facts']}")
        print(f"Discovery rate: {results['discovery_rate']:.1%}")
        print(f"\nEfficiency Metrics:")
        print(f"  Facts per turn: {results['facts_per_turn']:.3f}")
        print(f"  Facts per 1k tokens: {results['facts_per_1k_tokens']:.3f}")
        print(f"  Interviewer facts per 1k tokens: {results['interviewer_facts_per_1k_tokens']:.3f}")
        print(f"{'='*60}\n")

    return results


def main():
    """Main function for testing the evaluator."""
    import sys

    if len(sys.argv) > 1:
        session_path = sys.argv[1]
    else:
        # Default to first quirkbot session found
        sessions_dir = Path("data/sessions")
        session_files = list(sessions_dir.glob("*.json"))

        # Find a quirkbot session
        session_path = None
        for file in session_files:
            with open(file, 'r') as f:
                data = json.load(f)
                if data.get('metadata', {}).get('is_quirkbot') or 'quirkbot' in data.get('tags', []):
                    session_path = str(file)
                    print(f"Using QuirkBot session: {file.name}")
                    break

        if not session_path:
            print("No QuirkBot sessions found in data/sessions/")
            sys.exit(1)

    # Run evaluation
    results = evaluate_session(session_path, verbose=True)

    # Save results to JSON
    output_path = Path(session_path).stem + "_evaluation.json"
    with open(f"data/evaluations/{output_path}", 'w') as f:
        json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: data/evaluations/{output_path}")


if __name__ == "__main__":
    main()