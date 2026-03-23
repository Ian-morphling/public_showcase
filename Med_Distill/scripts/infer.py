"""
infer.py - Run inference on held-out test set for both base and fine-tuned models.
Saves all predictions to outputs/predictions.json for metrics.py and GitHub upload.
"""
 
import unsloth  # must be first import
import json
import argparse
from pathlib import Path
 
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
 
TEST_PATH        = Path("data/medquad_test.jsonl")
ADAPTER_PATH     = Path("outputs/qwen1.5b_medqa")
BASE_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
PREDICTIONS_PATH = Path("outputs/predictions.json")
MAX_NEW_TOKENS   = 256
 
SYSTEM_PROMPT = "You are a knowledgeable medical assistant. Answer clearly in plain language suitable for a patient."
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",      type=str, default=str(TEST_PATH))
    parser.add_argument("--adapter_path",   type=str, default=str(ADAPTER_PATH))
    parser.add_argument("--base_model",     type=str, default=BASE_MODEL)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--num_samples",    type=int, default=None)
    return parser.parse_args()
 
 
def load_quant_model(model_name: str, adapter_path: str = None):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model
 
 
def generate(tokenizer, model, question: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt    = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
 
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
 
 
def main():
    args = parse_args()
 
    with open(args.test_path, "r", encoding="utf-8") as f:
        test_rows = [json.loads(l) for l in f if l.strip()]
    if args.num_samples:
        test_rows = test_rows[:args.num_samples]
 
    questions  = [r["question"]       for r in test_rows]
    references = [r["teacher_answer"] for r in test_rows]
    print(f"Running inference on {len(test_rows)} test samples...\n")
 
    # ── Base model ─────────────────────────────────────────────────────────────
    print("Loading base model...")
    base_tok, base_model = load_quant_model(args.base_model)
    base_preds = [generate(base_tok, base_model, q, args.max_new_tokens)
                  for q in tqdm(questions, desc="Base model")]
    del base_model
    torch.cuda.empty_cache()
 
    # ── Fine-tuned model ───────────────────────────────────────────────────────
    print("\nLoading fine-tuned student...")
    ft_tok, ft_model = load_quant_model(args.base_model, args.adapter_path)
    ft_preds = [generate(ft_tok, ft_model, q, args.max_new_tokens)
                for q in tqdm(questions, desc="Fine-tuned")]
 
    # ── Save all predictions ───────────────────────────────────────────────────
    predictions = [
        {
            "question":         questions[i],
            "reference":        references[i],
            "base_output":      base_preds[i],
            "finetuned_output": ft_preds[i],
        }
        for i in range(len(test_rows))
    ]
 
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
 
    print(f"\nPredictions saved → {PREDICTIONS_PATH} ({len(predictions)} samples)")
    print("Next step: python scripts/metrics.py")
 
 
if __name__ == "__main__":
    main()