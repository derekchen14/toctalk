import re 
import os
import random

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

  for i, completion in enumerate(completions):
    # Extract the actual text content from the completion data structure
    completion_text = completion[0]["content"] if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
    
    # Debug logging - save sample completions to understand format
    if random.random() < 0.1:  # 10% chance to log
      os.makedirs("completion_samples", exist_ok=True)
      debug_file = os.path.join("completion_samples", "format_debug_samples.txt")
      with open(debug_file, "a") as f:
        f.write(f"\n\n=== COMPLETION {i} DEBUG ===\n")
        f.write(f"Raw completion type: {type(completion)}\n")
        f.write(f"Raw completion: {repr(completion)}\n")
        f.write(f"Extracted text: {repr(completion_text)}\n")
        f.write(f"Text length: {len(completion_text)}\n")
    
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    completion_text = "<think>" + completion_text
    # Check if the format is correct
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

    match = re.search(regex, completion_text, re.DOTALL) 
    # if the format is not correct, reward is 0
    if match is None or len(match.groups()) != 2:
      rewards.append(0.0)
      # Debug logging for failed matches
      if random.random() < 0.2:  # 20% chance to log failures
        os.makedirs("completion_samples", exist_ok=True)
        debug_file = os.path.join("completion_samples", "format_failures.txt")
        with open(debug_file, "a") as f:
          f.write(f"\n\n=== FORMAT FAILURE ===\n")
          f.write(f"Full text with synthetic <think>: {repr(completion_text)}\n")
          f.write(f"Regex used: {repr(regex)}\n")
          f.write(f"Match result: {match}\n")
    else:
      rewards.append(1.0)
      # Debug logging for successful matches
      if random.random() < 0.5:  # 50% chance to log successes
        os.makedirs("completion_samples", exist_ok=True)
        debug_file = os.path.join("completion_samples", "format_successes.txt")
        with open(debug_file, "a") as f:
          f.write(f"\n\n=== FORMAT SUCCESS ===\n")
          f.write(f"Full text: {repr(completion_text)}\n")
          f.write(f"Think content: {repr(match.group(1))}\n")
          f.write(f"Answer content: {repr(match.group(2))}\n")
  return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    Note: <think> is prefilled in the prompt, so completion should contain:
    reasoning</think>\n<answer>answer</answer>
    """
    import logging
    logger = logging.getLogger(__name__)

    # Debug: log first completion to understand structure
    if random.random() < 0.1:  # 10% of batches
        logger.info(f"\n=== FORMAT_REWARD DEBUG ===")
        logger.info(f"Completions type: {type(completions)}")
        logger.info(f"Completions length: {len(completions)}")
        logger.info(f"First completion type: {type(completions[0])}")
        logger.info(f"First completion: {completions[0]}")
        if isinstance(completions[0], list) and len(completions[0]) > 0:
            logger.info(f"First completion[0] type: {type(completions[0][0])}")
            logger.info(f"First completion[0]: {completions[0][0]}")
            if isinstance(completions[0][0], dict):
                logger.info(f"First completion[0] keys: {completions[0][0].keys()}")
                logger.info(f"First completion[0]['content']: {completions[0][0].get('content', 'NO CONTENT KEY')}")

    pattern = r".*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

from math_verify import LatexExtractionConfig, parse, verify
def accuracy_reward(completions, target=None, **kwargs):
  solutions = kwargs["solution"]
  completion_contents = [completion[0]["content"] for completion in completions]
  rewards = []
  for content, solution in zip(completion_contents, solutions):
    gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    if len(gold_parsed) != 0:
      try:
        rewards.append(float(verify(answer_parsed, gold_parsed)))
      except Exception:
        rewards.append(0.0)
    else:
      rewards.append(1.0)
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

def completion_length_penalty(completions, **kwargs):
  """
  Applies a penalty for completions longer than 250 characters.
  Penalizes 1% per character above 250.
  
  Args:
    completions (list): Generated outputs
  Returns:
    list[float]: Penalty scores (1.0 for length <= 250, decreasing for longer)
  """
  rewards = []
  for completion in completions:
    try:
      # Extract the actual text content from the completion data structure
      completion_text = completion[0]["content"] if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
      
      # Count characters in the completion
      length = len(completion_text)
      
      if length <= 250:
        rewards.append(1.0)  # No penalty
      else:
        # Apply 1% penalty per character over 250
        excess_chars = length - 250
        penalty = max(0.0, 1.0 - 0.01 * excess_chars)
        rewards.append(penalty)
        
      # Debug logging occasionally
      if random.random() < 0.05:  # 5% chance
        os.makedirs("completion_samples", exist_ok=True)
        debug_file = os.path.join("completion_samples", "length_penalty_debug.txt")
        with open(debug_file, "a") as f:
          f.write(f"\n=== LENGTH PENALTY DEBUG ===\n")
          f.write(f"Length: {length} chars\n")
          f.write(f"Penalty: {rewards[-1]:.3f}\n")
          f.write(f"Content: {repr(completion_text[:100])}...\n")
          
    except Exception as e:
      rewards.append(0.5)  # Default middle penalty if extraction fails
  return rewards

