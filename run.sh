#!/bin/bash
set -e
set -x

# ____ Start web application ____
# python web_chat.py --port 8020

# ____ Start CLI chat ____
# python terminal_chat.py --model_size medium

# ____ Start model training ____
python training.py --model_size small --lora_r 8 --lora_alpha 32 --learning_rate 1e-5 \
   --max_completion_length 256 --max_prompt_length 128 --method grpo --dataset_path "AI-MO/NuminaMath-TIR" \
   --num_generations 4 --logging_steps 10 --save_strategy "steps" --grad_accum_steps 16 \
   --lora_target_modules "q_proj,v_proj" --allow_download --per_device_bs 32
