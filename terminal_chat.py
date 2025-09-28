#!/usr/bin/env python3

import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils.arguments import parse_arguments
from utils.helpers import get_device_info, get_model_name

def get_personality_prompt():
    system_prompt = """You are Chuck, a charismatic and experienced used car salesman. You're friendly, persuasive, and always looking to make a deal. You have years of experience and know every trick in the book, but you're also genuinely helpful. You speak with enthusiasm, use car salesman lingo, and always try to find the perfect vehicle for your customers. You're quick with compliments, love to build rapport, and never take no for an answer too easily. Remember to ask about their needs, budget, and what they're looking for in a vehicle. Always end your responses by trying to move the conversation toward making a sale or getting them to look at a specific car.
    
    Speak in relatively concise sentences. Your goal is to understand the customer's needs rather than to force them into a sale.
    """
    return system_prompt

def load_model(args, device_info):
    """Load the specified model with optimal device configuration."""
    # Get the appropriate model name based on args and device
    model_name = get_model_name(args)
    
    if args.verbose:
        print(f"Loading model: {model_name}")
        print(f"Device info: {device_info}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token for Qwen models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or "[PAD]"
        
        # Prepare model loading arguments
        model_kwargs = {
            "dtype": device_info["torch_dtype"],  # Use 'dtype' instead of deprecated 'torch_dtype'
            "trust_remote_code": True,
        }
        
        # Add device configuration
        if device_info["device_map"]:
            model_kwargs["device_map"] = device_info["device_map"]
        device = device_info["device"]
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device if not using device_map
        if not model_kwargs.get("device_map") and device != "cpu":
            model = model.to(device)
        
        if args.verbose:
            print(f"Model loaded successfully on {device}")
            
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to CPU...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token for Qwen models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or "[PAD]"
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        return model, tokenizer, "cpu"

def generate_response(model, tokenizer, device, conversation_history, args):
    """Generate response using Qwen format with proper role formatting."""
    # Format conversation for Qwen models with role labels
    formatted_conversation = ""
    for entry in conversation_history:
        if entry["role"] == "system":
            formatted_conversation += f"System: {entry['content']}\n\n"
        elif entry["role"] == "user":
            formatted_conversation += f"Customer: {entry['content']}\n\n"
        elif entry["role"] == "assistant":
            formatted_conversation += f"Agent: {entry['content']}\n\n"
    
    formatted_conversation += "Agent: "
    
    # Tokenize input with attention mask
    inputs = tokenizer(formatted_conversation, return_tensors="pt", padding=True, truncation=True)
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    generation_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=20,  # Explicitly set to match Qwen default
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=151643,  # Explicitly set Qwen's default BOS token ID
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the new part (after the formatted conversation)
    response = full_response[len(formatted_conversation):].strip()
    
    # Clean up response (remove any potential continuation of conversation)
    if "Customer:" in response:
        response = response.split("Customer:")[0].strip()
    if "System:" in response:
        response = response.split("System:")[0].strip()
    if "Agent:" in response:
        response = response.split("Agent:")[0].strip()
        
    return response

def main():
    args = parse_arguments()
    
    # Get device information
    device_info = get_device_info()
    
    if args.verbose:
        print(f"Hardware detected: {device_info['hardware_type']}")
        print(f"Using device: {device_info['device']}")
    
    # Load model
    model_name = get_model_name(args)
    print(f"Loading model '{model_name}'... This may take a moment.")
    model, tokenizer, device = load_model(args, device_info)
    print("Model loaded successfully!")
    
    # Initialize conversation with system prompt
    system_prompt = get_personality_prompt()
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    print("\n" + "="*60)
    print("🚗 Welcome to Chuck's Used Car Emporium! 🚗")
    print("="*60)
    print("Type 'q' or 'quit' to exit")
    print("Type 'r' or 'restart' to start a new conversation")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['q', 'quit']:
                print("\nAgent: Thanks for stopping by! Come back anytime - I'll have the perfect car waiting for you!")
                break
            elif user_input.lower() in ['r', 'restart']:
                # Reset conversation with system prompt
                system_prompt = get_personality_prompt()
                conversation_history = [
                    {"role": "system", "content": system_prompt}
                ]
                print("\n" + "-"*60)
                print("🔄 Starting fresh conversation with Agent!")
                print("-"*60)
                continue
            elif not user_input:
                continue
            
            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response with timing
            print("\nAgent: ", end="", flush=True)
            start_time = time.time()
            response = generate_response(model, tokenizer, device, conversation_history, args)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Print response with timing
            print(f"{response} ({elapsed_time:.1f}s)")
            
            # Add assistant response to conversation
            conversation_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\nAgent: Hey, no problem! Come back anytime - I've got cars that practically sell themselves!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Agent: Sorry, I got a bit tongue-tied there. What were you saying?")

if __name__ == "__main__":
    main()