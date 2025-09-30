# QuirkBot Evaluation Framework

## Instructions for Using This Document

1. Work in sequence: Complete phases in order (Phase 0 → Phase 1 → ...)
2. Mark progress: Use `- [x]` to mark completed subtasks
3. Fill Developer Notes: Add notes at beginning and end of each phase as work progresses
4. Sanity checks: Complete ALL sanity checks before moving to next phase
5. Ask before deviating: If unexpected issues arise, ask permission to deviate from plan
6. Use `tmp_scripts/`: Create any temporary scripts/tests/configs here to keep the repo tidy

---

## Context & Goal

QuirkBot sessions have interviewees with secret `embedded_facts` that interviewers try to discover through conversation. We need a simple evaluation script that takes a QuirkBot session JSON as input and outputs scores showing how well the interviewer discovered the facts.

---

## Phase 0: Understand the Data

**Goal**: Figure out the structure of session JSONs and where to find the embedded facts and interviewer notes

### Developer Notes - Phase 0 Start

Starting analysis of QuirkBot session structure to understand data format.

### Tasks

- [x] Load and examine a QuirkBot session JSON from `data/sessions/`
- [x] Find where the embedded_facts are stored in the session
- [x] Find where the interviewer's notes/discoveries are stored
- [x] Count the number of conversation turns

### Sanity Checks - Phase 0

- [x] Can successfully load and parse session JSON
- [x] Located embedded_facts in the data
- [x] Located interviewer notes/summaries
- [x] Understand the conversation structure

### Developer Notes - Phase 0 End

**Key Findings:**

1. **Session Structure**: QuirkBot sessions are stored as JSON with:
   - `session_id`: Unique identifier
   - `messages`: Array of conversation messages
   - `metadata`: Contains QuirkBot-specific info including system prompt
   - `tags`: Includes "quirkbot" tag

2. **Embedded Facts Location**:
   - Found in `metadata.system_prompt` (for metadata-style sessions)
   - Or in first `system` message content (for system-message-style sessions)
   - Facts are clearly marked in the prompt after "IMPORTANT EMBEDDED FACTS:"
   - Each fact has `fact_text` and `category`

3. **Interviewer Notes**:
   - Interviewer agent uses `<note>` tags within responses to track discoveries
   - Notes can be parsed using `src/note_parser.py` utilities
   - Notes mark "QUIRKY FACT DISCOVERED" when finding embedded facts

4. **Conversation Structure**:
   - Messages have `role` (system/user/assistant), `content`, `timestamp`
   - Turns = user + assistant messages (excluding system)
   - Example session had 8 turns (4 user, 4 assistant)

---

## Phase 1: Build Simple Evaluator

**Goal**: Create a script that compares interviewer notes to embedded facts using an LLM and outputs basic scores

### Developer Notes - Phase 1 Start

Created evaluator module to analyze QuirkBot sessions with interviewer notes.

### Tasks

- [x] Create `src/quirkbot_evaluator.py`
- [x] Load session JSON and extract embedded_facts
- [x] Extract interviewer's final notes
- [x] Use LLM to compare notes against each fact (discovered: yes/no)
- [x] Calculate simple scores: facts_discovered / total_facts
- [x] Print results to console

### Sanity Checks - Phase 1

- [x] Script runs on a single session file
- [x] LLM comparison gives reasonable yes/no answers
- [x] Score calculation is correct (e.g., 3/5 facts = 0.6)
- [x] Output is clear and readable

### Developer Notes - Phase 1 End

**Implementation Details:**
- Created `quirkbot_evaluator.py` that uses ClaudeLLMClient to evaluate fact discovery
- Handles multiple session formats (metadata embedded_facts, system prompts)
- Uses note_parser to extract interviewer notes with `<note>` tags
- LLM evaluates each fact with JSON response format
- Generates clear console output and saves JSON results

**Testing:**
- Created `run_interviewer_session.py` to generate test data
- Successfully evaluated session with 100% discovery rate
- Output shows detailed fact-by-fact analysis plus summary metrics

---

## Phase 2: Add Rate Metrics

**Goal**: Calculate facts per turn and facts per token to measure efficiency

### Developer Notes - Phase 2 Start

Adding token counting and efficiency metrics to the evaluator.

### Tasks

- [x] Count conversation turns in the session
- [x] Count total tokens (approximate or use tiktoken)
- [x] Calculate facts_per_turn = facts_discovered / num_turns
- [x] Calculate facts_per_token = facts_discovered / total_tokens
- [x] Output all metrics in a clean JSON format

### Sanity Checks - Phase 2

- [x] Turn counting is accurate
- [x] Token counting gives reasonable numbers
- [x] Rate metrics make sense (not zero, not huge)
- [x] JSON output is valid and includes all metrics

### Developer Notes - Phase 2 End

**Implementation Details:**
- Added tiktoken for accurate token counting (with word-based fallback)
- Track tokens separately for interviewer and interviewee
- Calculate multiple efficiency metrics:
  - Facts per turn (0.077 in test)
  - Facts per 1k tokens overall (0.389 in test)
  - Interviewer facts per 1k tokens (0.798 in test)
- Enhanced JSON output with all token and efficiency metrics

**Test Results:**
- Successfully counted 2,572 total tokens
- Separated interviewer (1,253) and interviewee (1,319) tokens
- All metrics calculated correctly and saved to JSON
- Clean, readable output format for analysis