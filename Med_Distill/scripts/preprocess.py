"""
preprocess.py - Filter and sample lavita/MedQuAD for the distillation pipeline.
Keeps reference_answer alongside question so evaluate.py can do a 3-way comparison.
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset

OUTPUT_PATH      = Path("data/medquad_filtered.jsonl")
MIN_QUESTION_LEN = 10
MIN_ANSWER_LEN   = 50
MAX_ANSWER_LEN   = 2000  # prevents context overflow during teacher generation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def filter_dataset(dataset) -> list[dict]:
    filtered = []
    for row in dataset:
        q = (row.get("question") or "").strip()
        a = (row.get("answer")   or "").strip()
        if not a or len(q) < MIN_QUESTION_LEN:
            continue
        if len(a) < MIN_ANSWER_LEN or len(a) > MAX_ANSWER_LEN:
            continue
        filtered.append({"question": q, "reference_answer": a})
    return filtered


def main():
    args = parse_args()
    output_path = Path(args.output_path)

    print("Loading lavita/MedQuAD...")
    dataset  = load_dataset("lavita/MedQuAD", split="train")
    print(f"  Raw rows: {len(dataset):,}")

    filtered = filter_dataset(dataset)
    print(f"  After filtering: {len(filtered):,}")

    random.seed(args.seed)
    sampled = random.sample(filtered, min(args.sample_size, len(filtered)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item) + "\n")

    print(f"  Saved {len(sampled):,} samples → {output_path}")
    print("\nNext step: python scripts/generate_teacher.py")


if __name__ == "__main__":
    main()
