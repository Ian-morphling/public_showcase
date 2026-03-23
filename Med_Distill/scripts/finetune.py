"""
finetune.py - Fine-tune Qwen2.5-1.5B-Instruct on clean teacher-generated MedQuAD labels.
Splits clean data into train/test before training — test set saved for evaluate.py.
Uses Unsloth FastLanguageModel + TRL SFTTrainer with completion-only label masking.
"""
 
import json
import argparse
import random
from pathlib import Path
 
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
 
INPUT_PATH    = Path("data/medquad_teacher_clean.jsonl")
TEST_PATH     = Path("data/medquad_test.jsonl")
OUTPUT_DIR    = Path("outputs/qwen1.5b_medqa")
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN   = 1024
TEST_SIZE     = 0.1  # 10% held out for evaluation
 
SYSTEM_PROMPT = "You are a knowledgeable medical assistant. Answer clearly in plain language suitable for a patient."
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",       type=str,   default=str(INPUT_PATH))
    parser.add_argument("--output_dir",       type=str,   default=str(OUTPUT_DIR))
    parser.add_argument("--model_name",       type=str,   default=STUDENT_MODEL)
    parser.add_argument("--num_train_epochs", type=int,   default=3)
    parser.add_argument("--per_device_batch", type=int,   default=4)
    parser.add_argument("--grad_accum_steps", type=int,   default=4)
    parser.add_argument("--learning_rate",    type=float, default=2e-4)
    parser.add_argument("--seed",             type=int,   default=42)
    return parser.parse_args()
 
 
def split_and_save(rows: list[dict], test_size: float, seed: int, test_path: Path):
    """Split rows into train/test, save test set for evaluate.py."""
    random.seed(seed)
    random.shuffle(rows)
    split_idx  = int(len(rows) * (1 - test_size))
    train_rows = rows[:split_idx]
    test_rows  = rows[split_idx:]
 
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row) + "\n")
 
    print(f"  Train: {len(train_rows):,} | Test: {len(test_rows):,} → {test_path}")
    return train_rows
 
 
def load_model(model_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer
 
 
def format_sample(row: dict, tokenizer) -> dict:
    """Format as chat template — SFTTrainer masks prompt tokens via completion_only_loss."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": row["question"]},
        {"role": "assistant", "content": row["teacher_answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}
 
 
def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print("Loading clean dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"  Total clean rows: {len(rows):,}")
 
    # Split and save test set before training
    train_rows = split_and_save(rows, TEST_SIZE, args.seed, TEST_PATH)
 
    print(f"\nLoading student model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
 
    train_dataset = Dataset.from_list([format_sample(r, tokenizer) for r in train_rows])
 
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            optim="adamw_8bit",
            seed=args.seed,
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            completion_only_loss=True,
            report_to="none",
        ),
    )
 
    print("\nStarting training...")
    trainer.train()
 
    print(f"\nSaving adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Training complete.\nNext step: python scripts/evaluate.py")
 
 
if __name__ == "__main__":
    main()
 