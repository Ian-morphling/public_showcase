# python scripts/chat_qlora.py --max_new_tokens 64

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--adapter_path", type=str, default="outputs/qlora_opt_1.3b")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()

def load_model(model_name, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(base, adapter_path)
    return tokenizer, lora_model

def chat_loop(tokenizer, model, max_new_tokens=64):
    print("\n Music Review Summarizer Chatbot (QLoRA)")
    print("Type 'exit' to quit.\n")

    history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        history_text = "\n".join(history)
        prompt = f"{history_text}\nInstruction: Summarize this review.\nInput: {user_input}\nOutput:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Output:")[-1].strip()
        print(f"Chatbot: {response}\n")

        history.append(f"User: {user_input}")
        history.append(f"Chatbot: {response}")

def main():
    args = parse_args()
    tokenizer, model = load_model(args.model_name, args.adapter_path)
    chat_loop(tokenizer, model, args.max_new_tokens)

if __name__ == "__main__":
    main()
