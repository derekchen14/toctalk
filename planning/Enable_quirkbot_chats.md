# Enable Quirkbot Chats

## Instructions for Using This Document

1. Work in sequence: Complete phases in order (Phase 0 → Phase 1 → ...)
2. Mark progress: Use `- [x]` to mark completed subtasks
3. Fill Developer Notes: Add notes at beginning and end of each phase as work progresses
4. Sanity checks: Complete ALL sanity checks before moving to next phase
5. Ask before deviating: If unexpected issues arise, ask permission to deviate from plan
6. Use `tmp_scripts/`: Create any temporary scripts/tests/configs here to keep the repo tidy

---

## Context & Goal

The quirkbot task is a task-oriented conversation experiment where an LLM agent plays the role of a synthetic person based on biographical sketches. The goal is to enable chat sessions where the agent embodies one of these synthetic personas from a benchmark dataset, e.g. `benchmarks/quirkbot_benchmark_v0.json`. Each persona has a natural narrative biography and embedded "quirky" facts. The agent should know it's part of an experiment to measure task-oriented conversational skills and should naturally reveal these embedded facts when prompted through appropriate conversation topics. This v0 implementation will include all biographical information in the agent's prompt, with future versions potentially hiding some information for retrieval-based approaches.

---

## Phase 0: Minimal Quirkbot Implementation

**Goal**: Create the simplest possible working quirkbot chat by adding persona loading and a system prompt to the existing chat system.

### Developer Notes - Phase 0 Start

Starting Phase 0 implementation at 2025-09-21.
- Examined existing codebase structure (ChatManager, ChatSession, LLMClient)
- Reviewed benchmark file format with 45 personas containing embedded facts
- Plan: Create minimal quirkbot module and modify ChatManager to support system prompts

### Tasks

- [x] Create `src/quirkbot.py` with functions to load personas from JSON
- [x] Add a simple system prompt generator that includes persona bio and facts
- [x] Modify ChatManager to accept an optional system prompt parameter
- [x] Create `tmp_scripts/test_quirkbot.py` to test a basic quirkbot conversation

### Sanity Checks - Phase 0

- [x] Can load and parse the benchmark JSON file
- [x] System prompt includes both narrative and embedded facts
- [x] Test script successfully initiates a quirkbot conversation
- [x] Agent responds in character based on the persona

### Developer Notes - Phase 0 End

Phase 0 completed successfully.
- Created `src/quirkbot.py` with persona loading and system prompt generation
- Modified ChatManager to accept optional system_prompt parameter
- Fixed Python generator issue by splitting chat methods to avoid yield/return conflict
- Test script confirms agent embodies personas and naturally reveals embedded facts
- All 45 personas load correctly, facts emerge organically in conversation

---

## Phase 1: Add Web Interface Support

**Goal**: Expose quirkbot functionality through the existing web interface with minimal UI changes. We will reuse the current chat web app at http://localhost:8000, keeping all existing functionality intact.

### Developer Notes - Phase 1 Start

Starting Phase 1 implementation at 2025-09-21.
- Examined web app structure (FastAPI backend, vanilla JS frontend)
- Plan: Add quirkbot API endpoint and integrate with existing UI

### Tasks

- [x] Add `/api/quirkbot/random` endpoint to the existing FastAPI server in `src/web_app.py`
- [x] Add a simple "Start Quirkbot Chat" button to the existing `static/index.html`
- [x] Display current persona name/age in the existing chat header when in quirkbot mode
- [x] Store persona ID in session metadata (reusing existing session storage system)

### Sanity Checks - Phase 1

- [x] Button successfully starts a quirkbot chat with random persona
- [x] Persona information displays correctly in existing UI
- [x] Sessions save and reload with persona intact (using existing session system)
- [x] Regular chat mode still works normally without any breaking changes

### Developer Notes - Phase 1 End

Phase 1 completed successfully.
- Added `/api/quirkbot/random` endpoint that creates sessions with personas
- Modified chat endpoints to inject system prompt for quirkbot sessions
- Added purple "Start Quirkbot Chat" button to UI
- Updated session info display to show persona name/age for quirkbot sessions
- All tests pass - both regular and quirkbot modes work correctly
- Sessions persist with metadata intact including system prompts

---

## Phase 2: Testing and Polish

**Goal**: Ensure the quirkbot system works reliably and document usage.

### Developer Notes - Phase 2 Start

... developer notes BEFORE STARTING work goes here ...

### Tasks

- [ ] Test conversations with at least 5 different personas
- [ ] Add error handling for malformed benchmark files
- [ ] Update README.md with quirkbot usage instructions
- [ ] Create example conversation showing fact discovery

### Sanity Checks - Phase 2

- [ ] All 45 personas load without errors
- [ ] Facts emerge naturally in conversation
- [ ] Instructions are clear and complete
- [ ] System handles edge cases gracefully

### Developer Notes - Phase 2 End

... developer notes AFTER FINISHING work goes here ...