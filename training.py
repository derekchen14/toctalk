import logging
import os
import re
import torch

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig

from trl import SFTTrainer, ModelConfig, SFTConfig, GRPOTrainer, GRPOConfig
from datasets import load_dataset
from utils.helpers import get_checkpoint, make_conversation, get_device_info, get_model_name
from utils.arguments import parse_arguments
from utils.rewards import reasoning_format_reward, equation_reward_func, length_penalty_func, accuracy_reward
from utils.callbacks import RewardMetricsCallback

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def prepare_dataset(args):
  """Prepare dataset splits for training and evaluation."""
  # train_dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
  train_data, test_data = load_dataset(args.dataset_path, split=["train[:5%]", "test"])
  # train_dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

  # --- Can split further if we want ---
  # train_test_split = dataset.train_test_split(test_size=0.1)
  # train_dataset = train_test_split["train"]
  # train_dataset = train_dataset.shuffle(seed=42).select(range(4000))
  # print(train_data)
  train_dataset = train_data.map(make_conversation)
  train_dataset = train_dataset.remove_columns(["messages", "problem"])
  test_data = test_data.map(make_conversation)
  dataset_splits = {'train': train_dataset, 'dev': [], 'test': test_data}
  return dataset_splits

def prepare_model(args, peft_config, device_info):
  """ Load the model and tokenizer. """
  model_name = get_model_name(args)
  local_only = not args.allow_download
  print(f"device_info: {device_info}, model_name: {args.model_name}")

  if device_info['hardware_type'] == 'nvidia_gpu':
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    attention_implementation = 'flash_attention_2'
    torch_datatype = torch.bfloat16
    pin_memory = True

    quantization_config = BitsAndBytesConfig( load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoLigerKernelForCausalLM.from_pretrained(model_name,
      attn_implementation=attention_implementation,
      dtype=torch_datatype,
      device_map="auto",        # Hardcoded default
      use_cache=False,          # Needs to be turned off for training
      low_cpu_mem_usage=True,   # Should be True for both Apple Silicon and NVIDIA GPU
      quantization_config=quantization_config,
      local_files_only=local_only
    )
  else:
    use_liger = False
    attention_implementation = 'eager'
    torch_datatype = 'auto'
    pin_memory = False

    model = AutoModelForCausalLM.from_pretrained(model_name,
      attn_implementation=attention_implementation,
      dtype=torch_datatype,
      device_map="auto",
      use_cache=False,
      low_cpu_mem_usage=True,
      local_files_only=local_only
    )

  model = model.to(device_info['device'])
  peft_model = get_peft_model(model, peft_config)
  peft_model.print_trainable_parameters()

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  # Make sure not to include any special tokens in vocab, since the embedding layers are not trainable with PEFT
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  return peft_model, tokenizer

def run_training(args, model, training_config, dataset, tokenizer):
  """Main training function."""
  training_config.distributed_state.wait_for_everyone()  # wait for all processes to load
  
  # Initialize custom callback for enhanced logging
  reward_functions = [reasoning_format_reward, accuracy_reward] if args.method == "grpo" else []
  callback = RewardMetricsCallback(reward_functions=reward_functions)
  
  # Initialize the Trainer
  if args.method == "sft":
    trainer = SFTTrainer(model=model, args=training_config, train_dataset=dataset['train'],
        tokenizer=tokenizer, callbacks=[callback]
    )
  elif args.method == "grpo":
    trainer = GRPOTrainer(model=model, reward_funcs=reward_functions, args=training_config,
        train_dataset=dataset['train'], callbacks=[callback]
    )

  # Training loop
  last_checkpoint = get_checkpoint(training_config)
  if args.use_checkpoint and last_checkpoint is not None:
    logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')
    model_checkpoint = last_checkpoint
  else:
    model_checkpoint = None

  logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_config.num_train_epochs} epochs***')
  train_result = trainer.train(resume_from_checkpoint=model_checkpoint)

  metrics = train_result.metrics
  metrics['num_train_samples'] = len(dataset['train'])
  trainer.log_metrics('train', metrics)
  trainer.save_metrics('train', metrics)
  trainer.save_state()

  # Save model and create model card    
  logger.info('*** Save model ***')
  if trainer.is_fsdp_enabled and peft_config:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
  # Restore k,v cache for fast inference
  trainer.model.config.use_cache = True
  trainer.save_model(training_config.output_dir)
  logger.info(f'Model saved to {training_config.output_dir}')
  training_config.distributed_state.wait_for_everyone()  # wait for all processes to load

  tokenizer.save_pretrained(training_config.output_dir)
  logger.info(f'Tokenizer saved to {training_config.output_dir}')

  # Save everything else on main process
  # if trainer.accelerator.is_main_process:
  #   trainer.create_model_card({'tags': ['rl', 'quirkbot', 'toktalk']})
  # trainer.push_to_hub()  # if we want to push to Huggingface
  logger.info('*** Training complete! ***')

def run_evaluation(training_config, dataset, tokenizer):
  """Load the trained model and verify it's ready for inference."""
  logger.info("*** Loading trained model for evaluation ***")

  try:
    # Load the saved PEFT model from disk
    model = AutoPeftModelForCausalLM.from_pretrained(
        training_config.output_dir,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = True  # Enable KV cache for faster inference

    logger.info(f"✓ Model successfully loaded from {training_config.output_dir}")
    logger.info("✓ Model is ready for inference")

    # Optional: Show model info
    logger.info(f"Model type: {model.__class__.__name__}")
    logger.info(f"Device: {next(model.parameters()).device}")

  except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def args_to_configs(args, device_info):
  """Convert our unified args to ModelConfig and SFTConfig for TRL compatibility."""   
  # Parse target modules (handle comma-separated values)
  if args.lora_target_modules == "all-linear":
    target_modules = "all-linear"
  else:
    target_modules = [module.strip() for module in args.lora_target_modules.split(",")]
  peft_config = LoraConfig(task_type="CAUSAL_LM", target_modules=target_modules,
                      r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1)
  
  model_name = get_model_name(args)
  model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
  experiment_name = f"{model_name_short}_{args.model_size}_LR{args.learning_rate}_v{args.seed}"
  
  if args.output_dir.strip():
    use_tensorboard = True
    logging_dir = os.path.join(args.output_dir, experiment_name)
    report_to = ["tensorboard"]
  else:
    use_tensorboard = False
    logging_dir = None
    report_to = []

  if args.method == "sft":
    training_config = SFTConfig(
      output_dir=args.output_dir,
      num_train_epochs=args.num_train_epochs,
      per_device_train_batch_size=args.per_device_bs,
      gradient_accumulation_steps=args.grad_accum_steps,
      learning_rate=args.learning_rate,
      lr_scheduler_type=args.lr_scheduler_type,
      warmup_ratio=args.warmup_ratio,
      logging_steps=args.logging_steps,
      save_strategy=args.save_strategy,
      seed=args.seed,
      dataloader_pin_memory=device_info['pin_memory'],
      dataset_text_field="text",  # Hardcoded default
      packing=True,  # Hardcoded default
      report_to=report_to,
      logging_dir=logging_dir,
      run_name=experiment_name,
    )

  elif args.method == "grpo":
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        report_to=report_to, bf16=True, push_to_hub=False,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.logging_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        dataloader_pin_memory=device_info['pin_memory'],
        logging_dir=logging_dir,
        run_name=experiment_name,
    )

  
  return peft_config, training_config


if __name__ == '__main__':
  args = parse_arguments()
  device_info = get_device_info()
  peft_config, training_config = args_to_configs(args, device_info)
  set_seed(training_config.seed)

  dataset_splits = prepare_dataset(args)
  model, tokenizer = prepare_model(args, peft_config, device_info)
  
  run_training(args, model, training_config, dataset_splits, tokenizer)
  run_evaluation(training_config, dataset_splits, tokenizer)
