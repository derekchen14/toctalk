import os
import random
import torch
import platform
import requests

from trl import SFTConfig, GRPOConfig
from prompts.setup import *
from transformers.trainer_utils import get_last_checkpoint

def get_checkpoint(training_args: SFTConfig | GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def make_conversation(task):
  match task:
    case "math": return make_ioi_conversation
    case "countdown": return make_countdown_conversation
    case "quirkbot": return make_quirkbot_conversation
    case _: raise ValueError(f"Unknown task: {task}")
  
def make_quirkbot_conversation(example):
  convo = {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": example["question"]},
      {"role": "assistant", "content": example["answer"]}
    ]
  }
  return convo 

def make_ioi_conversation(example):
  convo = {
    "prompt": [
      {"role": "system", "content": ioi_math_prompt},
      {"role": "user", "content": example["problem"]},
      {"role": "assistant", "content": "<think>"},  # Prefill the opening tag
    ],
  }
  return convo

def make_countdown_conversation(example):
  numbers, target = example["nums"], example["target"]
  user_msg = f"Using the numbers {numbers}, create an equation that equals {target}." + countdown_user_prompt

  convo = {
    "prompt": [
      {"role": "system", "content": countdown_system_prompt},
      {"role": "user", "content": user_msg},
      {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},  # Prefill the opening tag
    ],
  }
  return convo

def save_completion_sample(completion):
  if random.random() < 0.1:  # 1% chance to write samples into a file
    os.makedirs("completion_samples", exist_ok=True)
    log_file = os.path.join("completion_samples", "completion_samples.txt")
    with open(log_file, "a") as f:
      f.write(f"\n\n==============\n")
      f.write(completion)

def get_device_info():
    # Check for NVIDIA GPU (A100 or similar)
    if torch.cuda.is_available():
      gpu_name = torch.cuda.get_device_name(0).lower()
      device_info = {"device": "cuda", "device_map": "auto", "torch_dtype": torch.float16,
                "hardware_type": "nvidia_gpu", "pin_memory": True}
            
    # Check for Apple Silicon (M2 and similar)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      system_info = platform.platform().lower()
      device_info = {"device": "mps", "device_map": None, "torch_dtype": torch.float16,
                "hardware_type": "apple_silicon", "pin_memory": False}
    
    # Check for CPU only
    else:
      device_info = {"device": "cpu", "device_map": None, "torch_dtype": torch.float32,
                "hardware_type": "unknown", "pin_memory": False}
    
    return device_info

def get_model_name(args):
    """
    Get the appropriate model name based on args and device capabilities.
    If model_name is specified, validate it exists on HF Hub then use it (overrides model_size).
    Otherwise, map model_size to specific models that work with transformers.
    """
    if args.model_name:
      url = f"https://huggingface.co/{args.model_name}"
      try:
        response = requests.head(url, timeout=5)
        if response.status_code != 200:
          raise ValueError(f"Model '{args.model_name}' not found on Hugging Face Hub. Please check the model name.")
      except (requests.RequestException, requests.Timeout):
        raise ValueError(f"Model '{args.model_name}' could not be found in time, please check your internet connection.")
      return args.model_name

    size_to_model = {
        "tiny": "Qwen/Qwen3-0.6B",
        "small": "Qwen/Qwen2.5-3B-Instruct",
        "medium": "Qwen/Qwen3-4B-Instruct-2507", 
        "large": "Qwen/Qwen3-30B-A3B-Instruct-2507"
    }
    
    return size_to_model[args.model_size]
