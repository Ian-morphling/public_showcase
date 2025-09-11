# Demo: python scripts/generate_qlora.py --num_samples 3 --max_new_tokens 64

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--adapter_path", type=str, default="outputs/qlora_opt_1.3b")
    parser.add_argument("--data_file", type=str, default="data/Digital_Music_5_sample.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=3)
    return parser.parse_args()

def load_models(base_model, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto"
    )

    lora = PeftModel.from_pretrained(base, adapter_path)
    return tokenizer, base, lora

def generate_response(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_prompt(instruction, review_text):
    return f"Instruction: {instruction}\nInput: {review_text}\nOutput:"

def main():
    args = parse_args()
    tokenizer, base_model, lora_model = load_models(args.model_name, args.adapter_path)

    print(f"Loading {args.num_samples} samples from {args.data_file}...")
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    samples = dataset.select(range(min(args.num_samples, len(dataset))))

    print("\n=== Base vs Fine-Tuned QLoRA Model ===\n")
    for i, ex in enumerate(samples):
        prompt = build_prompt(ex["instruction"], ex["input"])
        print(f"\n--- Sample {i+1} ---")
        print(f"Original Review: {ex['input'][:300]}...")
        print(f"Reference Summary: {ex['output']}")

        base_out = generate_response(base_model, tokenizer, prompt, args.max_new_tokens)
        lora_out = generate_response(lora_model, tokenizer, prompt, args.max_new_tokens)

        print("\nBase Model Output:")
        print(base_out.split("Output:")[-1].strip())
        print("\nFine-Tuned (QLoRA) Output:")
        print(lora_out.split("Output:")[-1].strip())

if __name__ == "__main__":
    main()