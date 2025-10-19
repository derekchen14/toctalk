import re
import os
import random
import json
import sys
# Add parent directory to path to import llm_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import ClaudeLLMClient

"""
All reward functions should return a list of floats, one for each completion.
"""

def notepad_format_reward(completions, target, **kwargs):
  """
  Format: <think>...</think><note>...</note>
  Args:
    completions (list[str]): Generated outputs
    target (list[str]): Expected answers
  """
  rewards = []

  for completion, gt in zip(completions, target):
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    completion = "<think>" + completion
    # Check if the format is correct
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<note>([\s\S]*?)<\/note>$"

    match = re.search(regex, completion, re.DOTALL) 
    # if the format is not correct, reward is 0
    if match is None or len(match.groups()) != 2:
      rewards.append(0.0)
    else:
      rewards.append(1.0)
  return rewards


def reasoning_format_reward(completions, target=None, **kwargs):
  """
  Format: <think>...</think><answer>...</answer>
  Args:
    completions (list[str]): Generated outputs
    target (list[str]): Expected answers (optional, not used in format checking)
  """
  rewards = []

  for completion in completions:
    # Extract the actual text content from the completion data structure
    completion_text = completion[0]["content"] if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    completion_text = "<think>" + completion_text
    # Check if the format is correct
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

    match = re.search(regex, completion_text, re.DOTALL) 
    # if the format is not correct, reward is 0
    if match is None or len(match.groups()) != 2:
      rewards.append(0.0)
    else:
      rewards.append(1.0)
  return rewards

def reasoning_format_reward_alternate(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list

# from math_verify import LatexExtractionConfig, parse, verify
def accuracy_reward(completions, target=None, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # For now, return uniform rewards since math_verify is not imported
    # TODO: Implement proper accuracy checking when math_verify is available
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = [1.0] * len(completion_contents)  # Default reward for all completions
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
  """
  Evaluates completions based on:
  2. Mathematical correctness of the answer

  Args:
    completions (list[str]): Generated outputs
    target (list[str]): Expected answers
    nums (list[str]): Available numbers
  
  Returns:
    list[float]: Reward scores
  """
  rewards = []
  for completion, gt, numbers in zip(completions, target, nums):
   try:
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    completion = "<think>" + completion
    # Check if the format is correct
    match = re.search(r"<answer>(.*?)<\/answer>", completion)
    if match is None:
      rewards.append(0.0)
      continue
    # Extract the "answer" part from the completion
    equation = match.group(1).strip()
    # Extract all numbers from the equation
    used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
    
    # Check if all numbers are used exactly once
    if sorted(used_numbers) != sorted(numbers):
      rewards.append(0.0)
      continue
    # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
    allowed_pattern = r'^[\d+\-*/().\s]+$'
    if not re.match(allowed_pattern, equation):
      rewards.append(0.0)
      continue
    
    # Evaluate the equation with restricted globals and locals
    result = eval(equation, {"__builtins__": None}, {})
    # Check if the equation is correct and matches the ground truth
    if abs(float(result) - float(gt)) < 1e-5:
      rewards.append(1.0)
      if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
        os.makedirs("completion_samples", exist_ok=True)
        log_file = os.path.join("completion_samples", "success_completion_samples.txt")
        with open(log_file, "a") as f:
          f.write(f"\n\n==============\n")
          f.write(completion)
    else:
      rewards.append(0.0)
   except Exception:
      # If evaluation fails, reward is 0
      rewards.append(0.0) 
  return rewards

def length_penalty_func(completions, target, **kwargs):
  """
  Applies a length penalty to the rewards based on the length of the completion.
  Args:
    completions (list[str]): Generated outputs
    target (list[str]): Expected answers
  Returns:
    list[float]: Reward scores
  """
  rewards = []
  for completion, gt in zip(completions, target):
    try:
      # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
      completion = "<think>" + completion
      # Check if the format is correct
      match = re.search(r"<answer>(.*?)<\/answer>", completion)
      if match is None:
        rewards.append(0.0)
        continue
      # Extract the "answer" part from the completion
      answer = match.group(1).strip()
      # Apply length penalty
      rewards.append(1.0 / (len(answer) + 1))
    except Exception:
      rewards.append(0.0)
  return rewards


def quirkbot_fact_discovery(completions, target, **kwargs):
  """
  Evaluates whether interviewer notes successfully discover biographical facts.
  Uses Claude Haiku to semantically match extracted notes against target facts.

  Args:
    completions (list[dict]): List of conversation messages with role, content, timestamp
    target (list[dict]): List of biographical facts to discover, each with:
      - fact_id (int): Unique identifier
      - fact_text (str): Short description of the fact
      - category (str): Category of the fact
      - fact_narrative (str): Detailed narrative of the fact

  Returns:
    list[float]: Reward scores (1.0 for discovered fact, 0.0 otherwise)
  """
  rewards = []

  # Initialize Claude Haiku client for semantic matching
  try:
    llm_client = ClaudeLLMClient(model="claude-3-5-haiku-20241022")
  except Exception as e:
    print(f"Error initializing LLM client: {e}")
    # Return zeros if we can't initialize the client
    return [0.0] * len(completions)

  # Extract all notes from each completion message
  for completion in completions:
    # Handle both string and dict formats
    if isinstance(completion, dict):
      content = completion.get("content", "")
    else:
      content = str(completion)

    # Extract all <note>...</note> tags from the content
    note_matches = re.findall(r"<note>(.*?)<\/note>", content, re.DOTALL)

    # If no notes found, reward is 0.0
    if not note_matches:
      rewards.append(0.0)
      continue

    # For each note, check if it matches any target fact using LLM
    note_discovered_fact = False
    for note in note_matches:
      note = note.strip()
      if not note:
        continue

      # Create prompt for LLM to evaluate semantic match
      prompt = _create_fact_matching_prompt(note, target)

      try:
        messages = [
          {"role": "user", "content": prompt}
        ]

        response = llm_client.chat(messages, temperature=0.0, max_tokens=500)

        # Parse JSON response
        result = json.loads(response)

        # Check if any fact was discovered
        if result.get("match_found", False):
          note_discovered_fact = True
          break  # One discovered fact per completion is enough for reward

      except Exception as e:
        print(f"Error evaluating note with LLM: {e}")
        continue

    # Assign reward based on whether any note discovered a fact
    rewards.append(1.0 if note_discovered_fact else 0.0)

  return rewards


def _create_fact_matching_prompt(note, target_facts):
  """
  Creates a prompt for Claude to evaluate if a note matches any target facts.

  Args:
    note (str): The extracted note content
    target_facts (list[dict]): List of target biographical facts

  Returns:
    str: JSON-formatted prompt for the LLM
  """
  # Format target facts for the prompt
  facts_list = []
  for fact in target_facts:
    facts_list.append({
      "fact_id": fact["fact_id"],
      "fact_text": fact["fact_text"],
      "fact_narrative": fact["fact_narrative"]
    })

  prompt_data = {
    "task": "Evaluate whether the interviewer's note has successfully discovered any of the target biographical facts. A note 'discovers' a fact if it captures the key essence of that fact, even if worded differently.",
    "note": note,
    "target_facts": facts_list,
    "instructions": [
      "Compare the note against each target fact",
      "Consider semantic similarity, not exact word matching",
      "A discovery means the note captures the core information of the fact",
      "Return match_found=true if ANY fact is discovered",
      "Return the fact_id of the first matched fact (if any)",
      "Be reasonably strict - the note should clearly convey the unusual/quirky aspect of the fact"
    ],
    "output_format": {
      "match_found": "boolean - true if note matches any fact",
      "matched_fact_id": "integer or null - the fact_id of matched fact",
      "reasoning": "string - brief explanation of the decision"
    }
  }

  prompt = f"""You are evaluating whether an interviewer's note has successfully discovered a biographical fact.

INPUT:
{json.dumps(prompt_data, indent=2)}

OUTPUT (respond with valid JSON only):
"""

  return prompt

