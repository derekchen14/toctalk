import logging
import os
import re
import torch

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
# from liger_kernel.transformers import AutoLigerKernelForCausalLM

from trl import SFTTrainer, ModelConfig, SFTConfig, GRPOTrainer, GRPOConfig
from datasets import load_dataset
from utils.helpers import get_checkpoint, make_conversation, get_device_info, get_model_name
from utils.arguments import parse_arguments
from utils.rewards import reasoning_format_reward, equation_reward_func, length_penalty_func, accuracy_reward

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

def prepare_model_and_tokenizer(args):
  """ Load the model and tokenizer. """
  device_info = get_device_info()
  model_name = get_model_name(args)
  print(f"device_info: {device_info}, model_name: {args.model_name}")

  if device_info['hardware_type'] == 'nvidia_gpu':
    attention_implementation = 'flash_attention_2'
    torch_datatype = torch.bfloat16
  else:
    attention_implementation = 'eager'
    torch_datatype = 'auto'
  
  model_kwargs['quantization_config'] = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
      bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
  )
  
  # model = AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
  model = AutoModelForCausalLM.from_pretrained(model_name,
    attn_implementation=attention_implementation,
    dtype=torch_datatype,
    device_map="auto",        # Hardcoded default
    use_cache=False,          # Needs to be turned off for training
    low_cpu_mem_usage=True,   # Should be True for both Apple Silicon and NVIDIA GPU
  )
  # if the model does not have this method, it will raise an error, which is a good warning
  model.print_trainable_parameters()

  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  # Make sure not to include any special tokens in vocab, since the embedding layers are not trainable with PEFT
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  return model, tokenizer

def run_training(args, model, model_args, training_args, dataset, tokenizer):
  """Main training function."""
  # logger.info(f'Training/evaluation parameters {training_args}')

  training_args.distributed_state.wait_for_everyone()  # wait for all processes to load
  # Parse target modules (handle comma-separated values)
  if args.lora_target_modules == "all-linear":
    target_modules = "all-linear"
  else:
    target_modules = [module.strip() for module in args.lora_target_modules.split(",")]
  
  peft_config = LoraConfig(task_type="CAUSAL_LM", 
                           r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
                          target_modules=target_modules)

  # Initialize the Trainer
  if args.method == "sft":
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset['train'],
        tokenizer=tokenizer, peft_config=peft_config
    )
  elif args.method == "grpo":
    trainer = GRPOTrainer(model=model, reward_funcs=[reasoning_format_reward, accuracy_reward],
        args=training_args, train_dataset=dataset['train'], peft_config=peft_config
    )

  # set the device map for the model
  model = model.to(training_args.device)
  if trainer.accelerator.is_main_process and peft_config:
    trainer.model.print_trainable_parameters()

  # Training loop
  last_checkpoint = get_checkpoint(training_args)
  if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

  logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
  train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

  metrics = train_result.metricsf
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
  trainer.save_model(training_args.output_dir)
  logger.info(f'Model saved to {training_args.output_dir}')
  training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

  tokenizer.save_pretrained(training_args.output_dir)
  logger.info(f'Tokenizer saved to {training_args.output_dir}')

  # Save everything else on main process
  # if trainer.accelerator.is_main_process:
  #   trainer.create_model_card({'tags': ['rl', 'quirkbot', 'toktalk']})
  # trainer.push_to_hub()  # if we want to push to Huggingface
  logger.info('*** Training complete! ***')

def run_evaluation(model_args: ModelConfig, training_args: SFTConfig, dataset, tokenizer):
  """Main evaluation function."""
  logger.info("*** Running evaluation ***")

  # Load the saved model from disk
  model = AutoPeftModelForCausalLM.from_pretrained(
      training_args.output_dir,
      device_map="auto",
      torch_dtype=torch.bfloat16,
  )
  model.config.use_cache = True  # Enable KV cache for faster inference

  # Initialize evaluation trainer
  eval_trainer = SFTTrainer(
      model=model,
      args=training_args,
      train_dataset=None,  # Not needed for evaluation
      eval_dataset=dataset['dev'],
      tokenizer=tokenizer,
  )

  metrics = eval_trainer.evaluate()  
  # Log metrics
  eval_trainer.log_metrics("eval", metrics)
  eval_trainer.save_metrics("eval", metrics)
  logger.info(f"Evaluation results: {metrics}")

  # Optional: Run inference on test examples
  if len(dataset['test']) > 0:
      logger.info("*** Running inference on test set ***")
      predictions = eval_trainer.predict(dataset['test'])
      logger.info(f"Test results: {predictions.metrics}")

def args_to_configs(args):
  """Convert our unified args to ModelConfig and SFTConfig for TRL compatibility.""" 
  logger.info(args)
  

  model_args = ModelConfig(
    model_name_or_path=model_name,
    attn_implementation=attention_implementation,
    torch_dtype="auto",  # Let TRL handle this
    use_peft=True,
    load_in_4bit=args.load_in_4bit,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_target_modules=args.lora_target_modules,
  )
  
  if args.method == "sft":
    training_args = SFTConfig(
      output_dir=args.output_dir,
      num_train_epochs=args.num_train_epochs,
      per_device_train_batch_size=args.per_device_train_batch_size,
      gradient_accumulation_steps=args.grad_accum_steps,
      learning_rate=args.learning_rate,
      lr_scheduler_type=args.lr_scheduler_type,
      warmup_ratio=args.warmup_ratio,
      logging_steps=args.logging_steps,
      save_strategy=args.save_strategy,
      seed=args.seed,
      dataset_text_field="text",  # Hardcoded default
      packing=True,  # Hardcoded default
    )

  elif args.method == "grpo":
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        report_to=[], bf16=True, push_to_hub=False,  # Disable tensorboard
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.logging_steps,
        temperature=args.temperature,
        top_p=args.top_p,
    )

  
  return model_args, training_args


if __name__ == '__main__':
  args = parse_arguments()
  model_args, training_args = args_to_configs(args)
  set_seed(training_args.seed)

  dataset_splits = prepare_dataset(args)
  model, tokenizer = prepare_model_and_tokenizer(model_args, training_args)
  
  run_training(args, model, model_args, training_args, dataset_splits, tokenizer)
  run_evaluation(model_args, training_args, dataset_splits, tokenizer)
