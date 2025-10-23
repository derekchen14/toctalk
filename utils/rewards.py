import re 
import os
import random

"""
All reward functions should return a list of floats, one for each completion.
Input arguments:
  completions (list): Generated outputs
  target (list): Expected answers
  source (list): Inputs to the model, if applicable
"""

def notepad_format_reward(completions, target, source=None):
  """
  Format: <think>...</think><note>...</note>
    But the note tag is optional.
  Args:
    completions (list[str]): Generated outputs
    target (list[str]): Expected answers
  """
  rewards = []

  for completion in completions:
    try:
      content = '<think>' + completion[0]['content']
      regex = r"<think>(.*?)</think>\s*<note>(.*?)</note>"
      match = re.search(regex, content, re.DOTALL) 
      if match is None or len(match.groups()) > 2:
        rewards.append(0.0)
      else:
        rewards.append(1.0)
    except Exception:
      rewards.append(0.0)
  return rewards

def format_reward(completions, target, source=None):
  """Reward function that checks if the completion has a specific format.
  Note: <think> is prefilled in the prompt, so completion should contain:
  reasoning</think>\n<answer>answer</answer>
  """
  rewards = []
  for completion, ground_truth in zip(completions, target):

    try:
      content = '<think>' + completion[0]['content']
      if random.random() < 0.1:  # 10% of batches
        print(f"Sample: {content}")
        print(f"Target: {ground_truth}")
      regex_pattern = r'^<think>((?:(?!<think>|</think>).)*?)</think>\s*<answer>(.*?)</answer>$'

      match = re.search(regex_pattern, content, re.DOTALL)
      if match is None or len(match.groups()) != 2:
        rewards.append(0.0)
      else:
        rewards.append(1.0)
    except Exception:
      rewards.append(0.0)
  return rewards
 
def equation_reward(completions, target, source=None):
    """ Evaluates completions based on Mathematical correctness of the answer for Countdown Game
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        source (list[str]): Available numbers for forming the equation
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, ground_truth, numbers in zip(completions, target, source):
      try:
        content = completion[0]['content']
        # extract the answer part
        match = re.search(r"<answer>(.*?)<\/answer>", content)
        if match is None:
          rewards.append(0.0)
          continue

        # Extract all numbers from the equation
        equation = match.group(1).strip()
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
        if random.random() < 0.1:  # 10% of batches
          print("equation result:", result)
          print("ground truth:", ground_truth)

        if abs(float(result) - float(ground_truth)) < 1e-5:
          rewards.append(1.0)
        else:
          rewards.append(0.0)

      except Exception:
        rewards.append(0.0) 
    return rewards

from math_verify import LatexExtractionConfig, parse, verify
def accuracy_reward(completions, target, source=None):
  rewards = []
  for completion, solution in zip(completions, target):
    content = completion[0]['content']
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

def length_penalty(completions, target, source=None):
  """
  Applies a penalty for completions longer than 250 characters.
  Penalizes 1% per character above 250.
  """
  rewards = []
  for completion in completions:
    try:
      # Extract the actual text content from the completion data structure
      content = completion[0]["content"]
      length = len(content)
      
      if length <= 250:
        rewards.append(1.0)  # No penalty
      else:
        # Apply 1% penalty per character over 250
        excess_chars = length - 250
        penalty = max(0.0, 1.0 - 0.01 * excess_chars)
        rewards.append(penalty)
          
    except Exception as e:
      rewards.append(0.5)  # Default middle penalty if extraction fails
  return rewards

if __name__ == "__main__":
  correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7
  exactly once, with basic arithmetic operations, that equals 65. One possible
  combination is 55 + 36 - 19 + 7... </think>
  <answer> 55 + 36 - 7 - 19 </answer>"""
  correct_sample_2 = """ ... </think>
  <answer> 55 + 36 - 7 - 19 </answer>"""
  wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""
  wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
  95 + 88 = 183                                                                                                              
  Now, let's subtract 104 from 183 to get 79:
  183 - 104 = 79
  <think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""
  wrong_result = """ ... </think>
  <answer> 55 + 36 - 7 - 18 </answer>"""
  content = [correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result]
  completions = [[{"content": c}] for c in content]

  test_format_func = format_reward(completions, target=["65", "65", "65", "65", "65"], source=[[19, 36, 55, 7]] * 5)
  if test_format_func == [1.0, 1.0, 0.0, 0.0, 1.0]:
    print("Format Reward function is working!")
  else:
    print("Reward function for format is broken")

  test_equation_func = equation_reward(completions, target=["65", "65", "65", "65", "65"], source=[[19, 36, 55, 7]] * 5)
  if test_equation_func == [1.0, 1.0, 0.0, 0.0, 0.0]:
    print("Equation Reward function is working!")
  else:
    print("Reward function for equation is broken")