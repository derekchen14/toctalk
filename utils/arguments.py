import argparse

def parse_arguments():
  parser = argparse.ArgumentParser(description="Interactive chat system")
  
  # Model arguments
  model_group = parser.add_argument_group('model', 'Model configuration arguments')
  model_group.add_argument("--model_name", type=str, default=None, help="Model name or path using Huggingface Hub")
  model_group.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large"],
                           help="Model size, which will be overridden by model_name if provided")
  model_group.add_argument("--use_liger", action="store_true", help="Use Liger optimizations") 
 
  # Dataset arguments  
  dataset_group = parser.add_argument_group('dataset', 'Dataset and generation arguments')
  dataset_group.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
  dataset_group.add_argument("--max_completion_length", type=int, default=256, help="Maximum sequence length")
  dataset_group.add_argument("--max_prompt_length", type=int, default=512, help="Maximum sequence length")
  dataset_group.add_argument("--num_generations", type=int, default=8, help="Number of generations per prompt")
  dataset_group.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
  dataset_group.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
  dataset_group.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
  
  # PEFT arguments
  peft_group = parser.add_argument_group('peft', 'Parameter-Efficient Fine-Tuning arguments')
  peft_group.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
  peft_group.add_argument("--lora_target_modules", type=str, default="all-linear", help="LoRA target modules (comma-separated or 'all-linear')")
  peft_group.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
  peft_group.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
  
  # Training arguments
  training_group = parser.add_argument_group('training', 'Training configuration arguments')
  training_group.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
  training_group.add_argument("--method", type=str, default="sft", choices=["sft", "dpo", "ppo", "grpo"], help="Training method")
  training_group.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device")
  training_group.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
  training_group.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
  training_group.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type")
  training_group.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
  
  # Logging arguments
  logging_group = parser.add_argument_group('logging', 'Logging and output arguments')
  logging_group.add_argument("--seed", type=int, default=42, help="Random seed")
  logging_group.add_argument("--output_dir", type=str, default="./results", help="Output directory")
  logging_group.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
  logging_group.add_argument("--logging_steps", type=int, default=5, help="Logging frequency")
  logging_group.add_argument("--verbose", action="store_true", help="Enable verbose output")
  logging_group.add_argument("--port", type=int, default=8020, help="Port to run the server on")
  
  return parser.parse_args()