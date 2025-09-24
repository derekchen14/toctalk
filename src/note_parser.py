"""Utility functions for parsing notes from interviewer transcripts."""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def parse_notes_from_transcript(messages: List[Dict]) -> List[Dict]:
    """
    Extract all notes from a conversation transcript.

    Args:
        messages: List of message dictionaries with 'role', 'content', and optional 'timestamp'

    Returns:
        List of dictionaries containing:
            - 'note': The note content
            - 'turn_number': The message turn number (0-indexed)
            - 'timestamp': The message timestamp if available
            - 'role': The role who made the note (should be 'assistant' for interviewer)
    """
    notes = []
    note_pattern = r'<note>(.*?)</note>'

    for turn_number, message in enumerate(messages):
        role = message.get('role', '')
        content = message.get('content', '')
        timestamp = message.get('timestamp')

        # Find all note tags in the content
        matches = re.findall(note_pattern, content, re.DOTALL)

        for match in matches:
            note_text = match.strip()
            notes.append({
                'note': note_text,
                'turn_number': turn_number,
                'timestamp': timestamp,
                'role': role
            })

    return notes


def extract_notes_from_text(text: str) -> List[str]:
    """
    Extract all notes from a single text string.

    Args:
        text: Text containing potential <note> tags

    Returns:
        List of note contents
    """
    note_pattern = r'<note>(.*?)</note>'
    matches = re.findall(note_pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def strip_notes_from_text(text: str) -> str:
    """
    Remove all note tags and their content from text.

    Args:
        text: Text potentially containing <note> tags

    Returns:
        Text with all note tags removed
    """
    # Remove note tags and their content
    note_pattern = r'<note>.*?</note>'
    cleaned = re.sub(note_pattern, '', text, flags=re.DOTALL)

    # Clean up any double spaces left behind
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


def parse_notes_with_validation(text: str) -> Tuple[List[str], List[str]]:
    """
    Parse notes with validation for unclosed or malformed tags.

    Args:
        text: Text containing potential <note> tags

    Returns:
        Tuple of (valid_notes, errors)
            - valid_notes: List of successfully parsed note contents
            - errors: List of error messages for malformed notes
    """
    valid_notes = []
    errors = []

    # Check for unclosed opening tags
    open_tags = text.count('<note>')
    close_tags = text.count('</note>')

    if open_tags != close_tags:
        errors.append(f"Mismatched note tags: {open_tags} opening, {close_tags} closing")

    # Check for nested tags (simplified - just looks for pattern)
    if '<note>' in text:
        segments = text.split('<note>')
        for i, segment in enumerate(segments[1:], 1):
            if '<note>' in segment.split('</note>')[0] if '</note>' in segment else segment:
                errors.append(f"Potential nested note tags detected in segment {i}")

    # Extract valid notes
    note_pattern = r'<note>(.*?)</note>'
    matches = re.findall(note_pattern, text, re.DOTALL)
    valid_notes = [match.strip() for match in matches]

    # Check for empty notes
    for i, note in enumerate(valid_notes):
        if not note:
            errors.append(f"Empty note found at position {i + 1}")

    return valid_notes, errors


def format_notes_summary(notes: List[Dict]) -> str:
    """
    Format a list of parsed notes into a readable summary.

    Args:
        notes: List of note dictionaries from parse_notes_from_transcript

    Returns:
        Formatted string summary of all notes
    """
    if not notes:
        return "No notes found in transcript."

    summary_lines = ["Interview Notes Summary", "=" * 40]

    for i, note in enumerate(notes, 1):
        summary_lines.append(f"\nNote #{i} (Turn {note['turn_number']})")
        if note.get('timestamp'):
            summary_lines.append(f"Timestamp: {note['timestamp']}")
        summary_lines.append(f"Content: {note['note']}")

    summary_lines.append("\n" + "=" * 40)
    summary_lines.append(f"Total notes: {len(notes)}")

    # Count quirky facts discovered
    quirky_facts = [n for n in notes if 'QUIRKY FACT DISCOVERED' in n['note']]
    if quirky_facts:
        summary_lines.append(f"Quirky facts discovered: {len(quirky_facts)}")

    return '\n'.join(summary_lines)


def get_quirky_facts(notes: List[Dict]) -> List[str]:
    """
    Extract only the notes marked as quirky facts.

    Args:
        notes: List of note dictionaries

    Returns:
        List of quirky fact descriptions
    """
    quirky_facts = []

    for note in notes:
        content = note['note']
        if 'QUIRKY FACT DISCOVERED' in content:
            # Extract the fact description after the marker
            fact_match = re.search(r'QUIRKY FACT DISCOVERED[:\s]*(.*?)(?:\.|$)', content, re.IGNORECASE)
            if fact_match:
                quirky_facts.append(fact_match.group(1).strip())
            else:
                # Fallback: include the whole note if pattern doesn't match
                quirky_facts.append(content)

    return quirky_facts