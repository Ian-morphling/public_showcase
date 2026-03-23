"""
plot_loss.py - Generate training loss curve from saved trainer_state.json.
Saves chart to outputs/training_loss.png.
"""
 
import json
import argparse
from pathlib import Path
 
import matplotlib.pyplot as plt
 
TRAINER_STATE_PATH = Path("outputs/qwen1.5b_medqa/checkpoint-237/trainer_state.json")
OUTPUT_PATH        = Path("outputs/training_loss.png")
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_state", type=str, default=str(TRAINER_STATE_PATH))
    parser.add_argument("--output_path",   type=str, default=str(OUTPUT_PATH))
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    with open(args.trainer_state, "r", encoding="utf-8") as f:
        state = json.load(f)
 
    # Extract steps and loss values from log history
    log_history = state["log_history"]
    steps  = [e["step"]  for e in log_history if "loss" in e]
    losses = [e["loss"]  for e in log_history if "loss" in e]
    epochs = [e["epoch"] for e in log_history if "loss" in e]
 
    if not steps:
        print("No loss entries found in trainer_state.json.")
        return
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#4A90D9", linewidth=2, label="Training Loss")
 
    # Mark epoch boundaries with vertical lines
    seen_epochs = set()
    for step, epoch in zip(steps, epochs):
        epoch_int = int(epoch)
        if epoch_int > 0 and epoch_int not in seen_epochs and abs(epoch - epoch_int) < 0.05:
            ax.axvline(x=step, color="gray", linestyle="--", alpha=0.5)
            ax.text(step + 1, max(losses) * 0.98, f"Epoch {epoch_int}",
                    fontsize=8, color="gray")
            seen_epochs.add(epoch_int)
 
    ax.set_title("Training Loss — Qwen2.5-1.5B-Instruct QLoRA Fine-tuning\n"
                 "MedQuAD Knowledge Distillation (DeepSeek-R1-Distill-Qwen-7B → 1.5B)",
                 fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
 
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Loss curve saved → {output_path}")
 
 
if __name__ == "__main__":
    main()