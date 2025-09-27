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

from trl import SFTTrainer, ModelConfig, SFTConfig, get_peft_config
from datasets import load_dataset
from utils.helpers import get_checkpoint, create_conversation, get_device_info, get_model_name
from utils.arguments import parse_arguments
from utils.rewards import reasoning_format_reward, equation_reward_func, length_penalty_func

DATASET_PATH = 'OpenAssistant/oasst1'

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def prepare_dataset(dataset_path: str):
  """Prepare dataset splits for training and evaluation."""
  # train_dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
  # dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
  dataset = load_dataset(DATASET_PATH)
  # dev_dataset = load_dataset(DATASET_PATH, split="validation")
  # train_dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

  # --- Can split further if we want ---
  # train_test_split = dataset.train_test_split(test_size=0.1)
  # train_dataset = train_test_split["train"]
  # test_dataset = train_test_split["test"]
  print(dataset['train'][345].keys())    
  train_dataset = dataset.shuffle(seed=42).select(range(10000))
  logger.info(f'Loaded dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}')
  # dataset.to_json("train_dataset.json", orient="records")

  dataset_splits = {'train': dataset['train'], 'dev': dataset['validation'], 'test': []}
  return dataset_splits

def prepare_model_and_tokenizer(model_args: ModelConfig, training_args: SFTConfig):
  """ Load the model and tokenizer. """
  logger.info(f'Model parameters {model_args}')

  # define model kwargs
  if model_args.torch_dtype in ['auto', None]:
    torch_datatype = model_args.torch_dtype 
  else:
    torch_datatype = getattr(torch, model_args.torch_dtype)

  model_kwargs = dict(
      revision="main",  # Hardcoded default
      trust_remote_code=True,
      attn_implementation=model_args.attn_implementation, 
      torch_dtype=torch_datatype, 
      device_map="auto",  # Hardcoded default
      use_cache=False if training_args.gradient_checkpointing else True,
      low_cpu_mem_usage=False if torch.cuda.is_available() else True,
  )
  
  # Check which training method to use and if 4-bit quantization is needed
  if model_args.load_in_4bit: 
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )
  
  # load the model with our kwargs
  model_name = model_args.model_name_or_path
  # model = AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
  model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

  if hasattr(model, 'print_trainable_parameters'):
    model.print_trainable_parameters()

  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  # Make sure not to include any special tokens in vocab, since the embedding layers are not trainable with PEFT
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  return model, tokenizer

def run_training(model, model_args, training_args, dataset, tokenizer):
  """Main training function."""
  logger.info(f'Training/evaluation parameters {training_args}')

  training_args.distributed_state.wait_for_everyone()  # wait for all processes to load
  peft_config = get_peft_config(model_args)
  # alternate way to define peft config
  # peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1,
  #                         target_modules=["q_proj", "v_proj"])
  print(peft_config)

  # Initialize the Trainer
  trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset['train'],
      tokenizer=tokenizer, peft_config=peft_config
  )
  # trainer = GRPOTrainer(model=model_name, reward_funcs=[format_reward_func, equation_reward_func],
  #     args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['dev'], peft_config=peft_config
  # )

  if trainer.accelerator.is_main_process and peft_config:
    trainer.model.print_trainable_parameters()

  # Training loop
  last_checkpoint = get_checkpoint(training_args)
  if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

  logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
  train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

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
      torch_dtype=torch.float16,
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

def args_to_configs(args, device_info):
  """Convert our unified args to ModelConfig and SFTConfig for TRL compatibility."""
  
  # Get the appropriate model name
  model_name = get_model_name(args, device_info)
  
  # Create ModelConfig
  model_args = ModelConfig(
    model_name_or_path=model_name,
    tokenizer_name_or_path=args.tokenizer_name or model_name,
    attn_implementation=args.attn_implementation,
    torch_dtype="auto",  # Let TRL handle this
    use_peft=args.use_peft,
    load_in_4bit=args.load_in_4bit,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
  )
  
  # Create SFTConfig  
  training_args = SFTConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    save_strategy="epoch",  # Hardcoded default
    seed=args.seed,
    max_seq_length=args.max_seq_length,
    dataset_text_field="text",  # Hardcoded default
    packing=True,  # Hardcoded default
  )
  
  return model_args, training_args


if __name__ == '__main__':
  args = parse_arguments()
  device_info = get_device_info()
  model_args, training_args = args_to_configs(args, device_info)
  set_seed(training_args.seed)

  dataset_splits = prepare_dataset(DATASET_PATH)
  model, tokenizer = prepare_model_and_tokenizer(model_args, training_args)
  
  run_training(model, model_args, training_args, dataset_splits, tokenizer)
  run_evaluation(model_args, training_args, dataset_splits, tokenizer)
