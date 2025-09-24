import os
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
