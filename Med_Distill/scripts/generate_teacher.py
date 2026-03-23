"""
generate_teacher.py - Generate enriched answers using DeepSeek-R1-Distill-Qwen-7B.
Strips <think> blocks and saves only the final answer as the training label.
Supports resume — safely restart if interrupted without re-generating done rows.
"""

import re
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

INPUT_PATH    = Path("data/medquad_filtered.jsonl")
OUTPUT_PATH   = Path("data/medquad_teacher.jsonl")
TEACHER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
USER_PROMPT   = "You are a knowledgeable medical assistant. Answer clearly in plain language suitable for a patient.\n\n{question}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",     type=str, default=str(INPUT_PATH))
    parser.add_argument("--output_path",    type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--model_name",     type=str, default=TEACHER_MODEL)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def load_teacher(model_name: str):
    print(f"Loading teacher: {model_name}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 recommended for Blackwell sm_120
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": USER_PROMPT.format(question=question)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Force reasoning mode — R1-Distill skips <think> on simple prompts without this
    return prompt + "<think>\n"


def strip_think_block(text: str) -> str:
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: model didn't close the think block cleanly
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def generate_answer(tokenizer, model, question: str, max_new_tokens: int) -> str:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,  # DeepSeek-R1 recommended: 0.5–0.7
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens, not the prompt
    new_tokens = output_ids[0][input_len:]
    return strip_think_block(tokenizer.decode(new_tokens, skip_special_tokens=True))


def load_processed(output_path: Path) -> set[str]:
    """Return questions already written to output for resume support."""
    if not output_path.exists():
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        return {json.loads(l)["question"] for l in f if l.strip()}


def main():
    args = parse_args()
    input_path  = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    processed = load_processed(output_path)
    remaining = [r for r in rows if r["question"] not in processed]
    print(f"Total: {len(rows):,} | Done: {len(processed):,} | Remaining: {len(remaining):,}")

    if not remaining:
        print("All samples processed.")
        return

    tokenizer, model = load_teacher(args.model_name)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(remaining):
            teacher_answer = generate_answer(tokenizer, model, row["question"], args.max_new_tokens)
            out_f.write(json.dumps({
                "question":         row["question"],
                "reference_answer": row["reference_answer"],
                "teacher_answer":   teacher_answer,
            }) + "\n")
            out_f.flush()  # flush per row — safe against interruptions

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(remaining)}] processed")

    print(f"\nTeacher labels saved → {output_path}")
    print("Next step: python scripts/finetune.py")


if __name__ == "__main__":
    main()
