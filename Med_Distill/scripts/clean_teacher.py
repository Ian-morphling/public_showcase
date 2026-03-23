"""
clean_teacher.py - Remove rows where teacher_answer contains unstripped thinking traces.
Root cause: max_new_tokens=512 was too short for the teacher to close </think> tags.
"""
 
import json
import argparse
from pathlib import Path
 
INPUT_PATH  = Path("data/medquad_teacher.jsonl")
OUTPUT_PATH = Path("data/medquad_teacher_clean.jsonl")
 
# Thinking trace markers — answers starting with these are unstripped reasoning
THINK_MARKERS = [
    "Okay, so I need", "Let me think", "I think", "Let me ",
    "So, ", "Hmm,", "First,", "Step 1"
]
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, default=str(INPUT_PATH))
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_PATH))
    return parser.parse_args()
 
 
def is_dirty(answer: str) -> bool:
    return any(answer.startswith(m) for m in THINK_MARKERS)
 
 
def main():
    args = parse_args()
    input_path  = Path(args.input_path)
    output_path = Path(args.output_path)
 
    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
 
    clean_rows = [r for r in rows if not is_dirty(r.get("teacher_answer", ""))]
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in clean_rows:
            f.write(json.dumps(row) + "\n")
 
    print(f"Total    : {len(rows):,}")
    print(f"Removed  : {len(rows) - len(clean_rows):,}")
    print(f"Clean    : {len(clean_rows):,} → {output_path}")
    print("\nNext step: python scripts/finetune.py")
 
 
if __name__ == "__main__":
    main()