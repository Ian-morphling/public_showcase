"""
metrics.py - Compute ROUGE-1, ROUGE-2, ROUGE-L from saved predictions.
Loads outputs/predictions.json produced by infer.py.
Saves summary to outputs/eval_results.json.
"""
 
import json
import argparse
import importlib
from pathlib import Path
 
hf_evaluate = importlib.import_module("evaluate")
 
PREDICTIONS_PATH = Path("outputs/predictions.json")
RESULTS_PATH     = Path("outputs/eval_results.json")
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, default=str(PREDICTIONS_PATH))
    parser.add_argument("--results_path",     type=str, default=str(RESULTS_PATH))
    return parser.parse_args()
 
 
def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge        = hf_evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    return {
        "rouge_1": round(rouge_scores["rouge1"], 4),
        "rouge_2": round(rouge_scores["rouge2"], 4),
        "rouge_l": round(rouge_scores["rougeL"], 4),
    }
 
 
def main():
    args = parse_args()
 
    with open(args.predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
 
    print(f"Loaded {len(predictions)} predictions from {args.predictions_path}\n")
 
    references = [p["reference"]        for p in predictions]
    base_preds = [p["base_output"]      for p in predictions]
    ft_preds   = [p["finetuned_output"] for p in predictions]
 
    print("Computing ROUGE scores...")
    base_metrics = compute_metrics(base_preds, references)
    ft_metrics   = compute_metrics(ft_preds,   references)
 
    print("\n── Evaluation Results ──────────────────────────────────────────────")
    print(f"{'Metric':<20} {'Base Model':>12} {'Fine-tuned':>12}")
    print("-" * 46)
    for key in ["rouge_1", "rouge_2", "rouge_l"]:
        print(f"{key.upper():<20} {base_metrics[key]:>12} {ft_metrics[key]:>12}")
 
    results = {
        "num_test_samples": len(predictions),
        "base_model":       base_metrics,
        "finetuned_model":  ft_metrics,
        "sample_outputs":   predictions[:5],
    }
 
    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
 
    print(f"\nResults saved → {args.results_path}")
    print("Next step: python scripts/app.py")
 
 
if __name__ == "__main__":
    main()