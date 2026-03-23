"""
app.py - Streamlit demo for MedQA-Distill.
Shows: live inference (base vs fine-tuned), evaluation metrics, loss curve, sample outputs.
Run: streamlit run scripts/app.py
"""
 
import json
import sys
from pathlib import Path
 
import torch
import streamlit as st
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
 
# ── paths ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent.parent
ADAPTER_PATH     = ROOT / "outputs/qwen1.5b_medqa"
RESULTS_PATH     = ROOT / "outputs/eval_results.json"
PREDICTIONS_PATH = ROOT / "outputs/predictions.json"
TRAINER_STATE    = ROOT / "outputs/qwen1.5b_medqa/checkpoint-237/trainer_state.json"
BASE_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS   = 256
 
SYSTEM_PROMPT = "You are a knowledgeable medical assistant. Answer clearly in plain language suitable for a patient."
 
st.set_page_config(page_title="MedQA-Distill", page_icon="🩺", layout="wide")
 
 
# ── model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load both models once and cache for the session."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
 
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto"
    )
    base_model.eval()
 
    ft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(ft_model, str(ADAPTER_PATH))
    ft_model.eval()
 
    return tokenizer, base_model, ft_model
 
 
def generate(tokenizer, model, question: str) -> str:
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
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
 
 
# ── loss curve (plotly) ───────────────────────────────────────────────────────
def plot_loss_curve():
    if not TRAINER_STATE.exists():
        st.warning("trainer_state.json not found.")
        return
    with open(TRAINER_STATE) as f:
        state = json.load(f)
 
    log_history = state["log_history"]
    steps  = [e["step"]  for e in log_history if "loss" in e]
    losses = [e["loss"]  for e in log_history if "loss" in e]
    epochs = [e["epoch"] for e in log_history if "loss" in e]
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=losses, mode="lines",
        line=dict(color="#4A90D9", width=2),
        hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<br><extra></extra>",
        name="Training Loss"
    ))
 
    # Epoch boundary lines
    seen = set()
    for step, epoch in zip(steps, epochs):
        e_int = int(epoch)
        if e_int > 0 and e_int not in seen and abs(epoch - e_int) < 0.05:
            fig.add_vline(x=step, line_dash="dash", line_color="gray", opacity=0.5,
                          annotation_text=f"Epoch {e_int}", annotation_position="top")
            seen.add(e_int)
 
    fig.update_layout(
        title="Training Loss — Qwen2.5-1.5B QLoRA Fine-tuning",
        xaxis_title="Step", yaxis_title="Loss",
        hovermode="x unified", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
 
 
# ── main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("🩺 MedQA-Distill")
    st.caption(
        "Knowledge distillation from DeepSeek-R1-Distill-Qwen-7B → Qwen2.5-1.5B-Instruct "
        "fine-tuned on MedQuAD via QLoRA + Unsloth."
    )
 
    tab1, tab2, tab3 = st.tabs(["💬 Live Inference", "📊 Evaluation", "📉 Training Loss"])
 
    # ── Tab 1: Live Inference ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Ask a Medical Question")
        st.caption("Compare base Qwen2.5-1.5B vs fine-tuned student side by side.")
 
        question = st.text_area(
            "Enter a medical question:",
            placeholder="e.g. Is Trisomy 18 inherited?",
            height=80,
        )
 
        if st.button("Generate", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Loading models (first run takes ~30s)..."):
                    tokenizer, base_model, ft_model = load_models()
 
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Base Model** (Qwen2.5-1.5B-Instruct)")
                    with st.spinner("Generating..."):
                        base_out = generate(tokenizer, base_model, question)
                    st.info(base_out)
 
                with col2:
                    st.markdown("**Fine-tuned Student** (QLoRA + MedQuAD distillation)")
                    with st.spinner("Generating..."):
                        ft_out = generate(tokenizer, ft_model, question)
                    st.success(ft_out)
 
    # ── Tab 2: Evaluation ──────────────────────────────────────────────────────
    with tab2:
        st.subheader("Evaluation Results")
        st.caption("ROUGE scores on 141 held-out test samples. Reference: teacher-generated answers.")
 
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                results = json.load(f)
 
            base_m = results["base_model"]
            ft_m   = results["finetuned_model"]
 
            col1, col2, col3 = st.columns(3)
            metrics = [("ROUGE-1", "rouge_1"), ("ROUGE-2", "rouge_2"), ("ROUGE-L", "rouge_l")]
            for col, (label, key) in zip([col1, col2, col3], metrics):
                with col:
                    base_val = base_m[key]
                    ft_val   = ft_m[key]
                    pct      = round((ft_val - base_val) / base_val * 100, 1)
                    st.metric(
                        label=label,
                        value=f"{ft_val} (fine-tuned)",
                        delta=f"+{pct}% vs base ({base_val})",
                    )
 
            st.divider()
            st.markdown("**Sample Outputs** (from eval test set)")
            for i, sample in enumerate(results.get("sample_outputs", [])[:5]):
                with st.expander(f"Sample {i+1}: {sample['question'][:80]}..."):
                    st.markdown("**Question:**")
                    st.write(sample["question"])
                    st.markdown("**Reference (teacher answer):**")
                    st.write(sample["reference"])
                    st.markdown("**Base Model Output:**")
                    st.info(sample["base_output"])
                    st.markdown("**Fine-tuned Output:**")
                    st.success(sample["finetuned_output"])
        else:
            st.warning("eval_results.json not found. Run scripts/metrics.py first.")
 
    # ── Tab 3: Training Loss ───────────────────────────────────────────────────
    with tab3:
        st.subheader("Training Loss Curve")
        st.caption(
            "Loss across 3 epochs, 237 steps. "
            "Effective batch size: 16 (4 per device × 4 gradient accumulation). "
            "Optimizer: AdamW 8-bit with cosine LR decay."
        )
        plot_loss_curve()
 
 
if __name__ == "__main__":
    main()