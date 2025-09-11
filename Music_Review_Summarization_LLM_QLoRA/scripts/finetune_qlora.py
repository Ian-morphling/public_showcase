# Example: python scripts/finetune_qlora.py --num_train_epochs 1 --per_device_train_batch_size 2 --learning_rate 2e-4 --fp16

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--data_file", type=str, default="data/Digital_Music_5_sample.jsonl")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)  
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/qlora_opt_1.3b")
    return parser.parse_args()

def load_dataset_sft(file_path):
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    return dataset

def main():
    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset_sft(args.data_file)

    def tokenize_fn(example):
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
        tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    print("Loading base model with 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    print("Configuring QLoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    print("Starting training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit"  
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()

    print(f"Saving QLoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()