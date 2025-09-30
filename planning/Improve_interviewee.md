# Improve Interviewee Chat Agent Performance

## Instructions for Using This Document

1. Work in sequence: Complete phases in order (Phase 0 → Phase 1 → ...)
2. Mark progress: Use `- [x]` to mark completed subtasks
3. Fill Developer Notes: Add notes at beginning and end of each phase as work progresses
4. Sanity checks: Complete ALL sanity checks before moving to next phase
5. Ask before deviating: If unexpected issues arise, ask permission to deviate from plan
6. Use `tmp_scripts/`: Create any temporary scripts/tests/configs here to keep the repo tidy

---

## Context & Goal

The quirkbot task involves an interviewer agent and an interviewee agent engaged in a structured conversation. The interviewer's objective is to efficiently extract interesting ("quirky") biographical facts embedded within the interviewee's persona. The interviewee agent must simulate a natural interview conversation, playing the role of a specific person with given biographical details and embedded facts. Currently, the interviewee agent needs improvement to better simulate realistic interview behavior - answering only when asked, not volunteering information unnecessarily, and maintaining consistency while improvising details not explicitly provided. Our goal is to refine the interviewee's LLM prompt to achieve more natural, interview-appropriate responses while maintaining the challenge level for the interviewer agent training.

---

## Phase 0: Analyze Current Implementation & Define Requirements

**Goal**: Understand the existing interviewee implementation, identify weaknesses, and establish clear requirements for improvement.

### Developer Notes - Phase 0 Start

Found the interviewee prompt in `src/quirkbot.py:generate_system_prompt()`. The current implementation uses "natural_narrative" but in the test data it's "biographical_description".

Current prompt structure:
1. Provides full biography AND embedded facts to the interviewee
2. Instructions include "Let them emerge organically" but also "Be conversational!"
3. Has some good constraints but they're not strict enough

### Tasks

- [x] Locate and review the current interviewee agent prompt/implementation
- [x] Analyze sample conversation logs to identify specific behavioral issues
- [x] Document observed problems (e.g., volunteering info, small talk, asking questions back)
- [x] Define success criteria for improved interviewee behavior

### Success Criteria for Improved Interviewee

1. **Never asks questions back** to the interviewer about themselves
2. **Waits to be asked** before sharing information
3. **Answers concisely first**, then elaborates if follow-up questions are asked
4. **Does not volunteer embedded facts** - they must be discovered through proper questioning
5. **Maintains character** but acts like someone being interviewed, not a host
6. **Improvises consistently** when asked about details not in the biography
7. **Natural revelation** - embedded facts emerge through conversation, not as prepared speeches
8. **Text-only responses** - no action descriptions, just natural spoken words
- [x] Review the biographical data structure and embedded facts format

### Observed Problems from Sample Sessions

1. **Asking questions back to interviewer** (violates interview format):
   - Jacqueline: "Do you enjoy spending time outdoors?"
   - Elliot: "How are you doing today?"

2. **Over-volunteering information without being asked**:
   - Elliot immediately offers to "share some insights about technology, woodworking, or really any of my interests" without being asked
   - Jacqueline provides excessive detail about weekend activities unprompted

3. **Too eager to share embedded facts**:
   - Both personas reveal their quirky facts with minimal prompting
   - They provide long, detailed explanations rather than natural, gradual revelation

4. **Breaking interview format with unnecessary pleasantries**:
   - Offering to share insights
   - Acting like a host rather than an interviewee

5. **Data structure mismatch**:
   - Code expects "natural_narrative" but test data has "biographical_description"

6. **Using action descriptions** (discovered during live testing):
   - Interviewees include narrative actions like "*chuckles*" or "*pauses thoughtfully*"
   - This is inappropriate for a text-based chat interview format

### Sanity Checks - Phase 0

- [x] Current interviewee code/prompt has been located and understood
- [x] At least 3 sample conversations have been reviewed (2 full sessions analyzed)
- [x] Specific problematic behaviors have been documented with examples
- [x] Success criteria are clear and measurable

### Developer Notes - Phase 0 End

Phase 0 complete. Key findings:
- Current prompt is too permissive and encourages conversational behaviors inappropriate for an interview
- Interviewees are asking questions back and volunteering information
- Need stricter behavioral constraints
- Also found bug: code expects "natural_narrative" but data has "biographical_description"

---

## Phase 1: Design Improved Prompt Architecture

**Goal**: Create a comprehensive, clear prompt that instructs the interviewee agent to behave like a real person being interviewed, with proper constraints and guidance.

### Developer Notes - Phase 1 Start

Created improved prompt in `tmp_scripts/improved_interviewee_prompt.md` with:
- Clear role definition using intelligence operative analogy
- Strict behavioral rules (MUST/MUST NEVER sections)
- Proper interview dynamics explanation
- Separate sections for biography and embedded facts
- Improvisation guidelines

### Tasks

- [x] Draft core prompt structure with clear role definition
- [x] Add "intelligence operative legend" analogy to explain consistency requirements
- [x] Include explicit instructions about:
  - [x] Not volunteering embedded facts without being asked
  - [x] Not making small talk or asking questions back
  - [x] Improvising consistent details when needed
  - [x] Staying in character throughout
- [x] Add transparency about this being a research experiment for interviewer training
- [x] Structure prompt to include:
  - [x] Role definition section
  - [x] Behavioral constraints section
  - [x] Improvisation guidelines section
  - [x] Biographical information insertion point
  - [x] Embedded facts insertion point (for v0)

### Sanity Checks - Phase 1

- [x] Prompt clearly defines the interviewee's role
- [x] All problematic behaviors from Phase 0 are explicitly addressed
- [x] Instructions are unambiguous and actionable
- [x] Prompt structure allows easy insertion of biographical data

### Developer Notes - Phase 1 End

Phase 1 complete. Created comprehensive prompt that:
- Uses "MUST" and "MUST NEVER" for crystal-clear rules
- Explicitly forbids all problematic behaviors identified in Phase 0
- Provides clear mental model (intelligence operative with legend)
- Separates biography from embedded facts for clarity
- Ready for implementation in Phase 2

---

## Phase 2: Implement & Integrate Improved Prompt

**Goal**: Replace the existing interviewee prompt with the improved version and ensure proper integration with the system.

### Developer Notes - Phase 2 Start

Implementing the new prompt:
- Created backup at tmp_scripts/quirkbot_backup.py
- Fixed data structure bug (handles both 'biographical_description' and 'natural_narrative')
- Improved age extraction to handle various formats
- Added fact_narrative context when available

### Tasks

- [x] Create backup of current interviewee implementation
- [x] Implement the new prompt in the codebase
- [x] Ensure proper formatting and variable substitution for biographical data
- [x] Verify embedded facts are correctly included (for v0)
- [x] Test that the prompt integrates correctly with the existing chat flow

### Sanity Checks - Phase 2

- [x] Original implementation is backed up
- [x] New prompt compiles/runs without errors
- [x] Biographical data is correctly inserted into prompt
- [x] System can initiate a conversation with new prompt

### Developer Notes - Phase 2 End

Phase 2 complete. Implementation successful:
- New prompt properly integrated into generate_system_prompt()
- Fixed data structure compatibility issues
- Verified prompt generation with test script
- All key improvements verified in generated prompt
- Ready for interactive testing in Phase 3

---

## Phase 3: Interactive Testing with Human Interviewer

**Goal**: Conduct live testing sessions with John as the interviewer to evaluate and refine the interviewee agent's performance.

### Developer Notes - Phase 3 Start

During live testing, discovered additional issue:
- Interviewees were using action descriptions like "*chuckles*" and "*pauses thoughtfully*"
- Added explicit instructions in prompt to write text-only responses
- Added new "RESPONSE FORMAT" section with examples of correct vs wrong format
- Updated "MUST NEVER" rules to explicitly forbid action descriptions

### Tasks

- [ ] Set up test environment for manual interviewing sessions
- [ ] Conduct initial test conversation (5-10 exchanges)
- [ ] Document observed behaviors:
  - [ ] Does interviewee wait to be asked before sharing facts?
  - [ ] Are responses natural and in-character?
  - [ ] Is improvisation consistent with given biography?
  - [ ] Are embedded facts revealed appropriately?
- [ ] Identify areas needing adjustment
- [ ] Iterate on prompt based on observations
- [ ] Conduct follow-up test conversations after each adjustment

### Sanity Checks - Phase 3

- [ ] At least 3 test conversations completed
- [ ] Interviewee no longer volunteers information unprompted
- [ ] Responses feel natural and interview-appropriate
- [ ] Embedded facts are discoverable but not obvious
- [ ] John confirms improved behavior

### Developer Notes - Phase 3 End

... developer notes AFTER FINISHING work goes here ...

---

## Phase 4: Refinement & Documentation

**Goal**: Fine-tune the prompt based on testing results and document the improvements for future reference.

### Developer Notes - Phase 4 Start

... developer notes BEFORE STARTING work goes here ...

### Tasks

- [ ] Compile feedback from all test sessions
- [ ] Make final adjustments to prompt wording
- [ ] Test edge cases (e.g., very direct questions, vague questions)
- [ ] Document prompt design decisions and rationale
- [ ] Create comparison showing before/after conversation examples
- [ ] Update any relevant documentation or comments in code

### Sanity Checks - Phase 4

- [ ] Final prompt version tested successfully
- [ ] Documentation clearly explains the approach
- [ ] Before/after examples demonstrate improvement
- [ ] Code comments updated where relevant

### Developer Notes - Phase 4 End

... developer notes AFTER FINISHING work goes here ...

---

## Phase 5: Consider v1 Enhancement - Hidden Facts

**Goal**: Evaluate whether hiding embedded facts from the interviewee agent improves performance and plan implementation if beneficial.

### Developer Notes - Phase 5 Start

... developer notes BEFORE STARTING work goes here ...

### Tasks

- [ ] Test current implementation to see if agent volunteers facts despite instructions
- [ ] Design approach for hiding embedded facts while maintaining consistency
- [ ] Consider techniques like:
  - [ ] Providing only biographical description to interviewee
  - [ ] Using a separate fact-checking mechanism
  - [ ] Implementing post-processing to validate fact revelations
- [ ] Prototype and test hidden-facts approach
- [ ] Compare performance between v0 (facts visible) and v1 (facts hidden)
- [ ] Make recommendation for production approach

### Sanity Checks - Phase 5

- [ ] Both approaches tested with same biographical data
- [ ] Performance metrics compared objectively
- [ ] Recommendation backed by test results
- [ ] Implementation plan clear if v1 is chosen

### Developer Notes - Phase 5 End

... developer notes AFTER FINISHING work goes here ...