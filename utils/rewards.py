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

