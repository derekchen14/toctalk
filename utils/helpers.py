import os
import random
import torch
import platform
import requests

from trl import SFTConfig, GRPOConfig
from prompts.setup import system_message
from transformers.trainer_utils import get_last_checkpoint

def get_checkpoint(training_args: SFTConfig | GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def create_conversation(sample):
  convo = {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
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
      device_info = {"device": "cuda", "device_map": "auto", "torch_dtype": torch.float16, "hardware_type": "nvidia_gpu"}
      if "a100" in gpu_name:
        device_info["hardware_type"] = "a100"
            
    # Check for Apple Silicon (M2 and similar)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      system_info = platform.platform().lower()
      device_info = {"device": "mps", "device_map": None, "torch_dtype": torch.float16, "hardware_type": "apple_silicon"}
      if "arm64" in system_info:
        device_info["hardware_type"] = "m_series"
    
    # Check for CPU only
    else:
      device_info = {"device": "cpu", "device_map": None, "torch_dtype": torch.float32, "hardware_type": "unknown"}
    
    return device_info

def get_model_name(args, device_info):
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
        "small": "Qwen/Qwen3-0.6B",
        "medium": "Qwen/Qwen3-4B-Instruct-2507", 
        "large": "Qwen/Qwen3-30B-A3B-Instruct-2507"
    }
    
    return size_to_model[args.model_size]
