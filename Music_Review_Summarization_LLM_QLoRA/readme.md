# Music Review Summarization LLM Showcase

This project demonstrates **fine-tuning a Large Language Model (LLM)** using **QLoRA adapters** for summarizing Amazon Digital Music reviews. It showcases end-to-end skills in:

- Dataset preprocessing
- Instruction-following supervised fine-tuning (SFT)
- **QLoRA for memory-efficient 4-bit fine-tuning**
- Text generation and multi-turn chatbot demos
- Comparison of base vs fine-tuned model performance

Note: QLoRA enables **memory-efficient fine-tuning in 4-bit**, making it feasible to fine-tune larger LLMs on smaller GPUs without sacrificing performance. This project demonstrates this workflow on OPT-1.3B.


## Key skills demonstrated:

- Loading and preprocessing large-scale datasets for SFT
- Converting raw reviews into instruction-following prompts
- **Efficient memory-saving 4-bit fine-tuning using QLoRA**
- Tokenization, padding, and sequence preparation for causal LMs
- Supervised fine-tuning with `Trainer` and custom hyperparameters
- Generating and evaluating outputs from base vs QLoRA-adapted models
- Building a terminal-based interactive chatbot for demonstration
- Writing modular Python scripts suitable for deployment

## Dataset

**Amazon Digital Music 5-core reviews dataset** from [UCSD Amazon Review dataset collection](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).  

- **Sample size:** 5,000 reviews (for demonstration)  
- **Fields used:** `reviewText` and `summary`  
- Preprocessed into **instruction-following format** for supervised fine-tuning:

```json
{
  "instruction": "Summarize this review.",
  "input": "This album was amazing, but the last track was weak.",
  "output": "Good album, weak final track."
}
```

## Scripts Overview

### preprocess.py

- Loads gzipped Amazon Digital Music JSON dataset.
- Samples 5,000 reviews for demonstration.
- Converts reviews into **instruction-following format** (`instruction`, `input`, `output`).
- Saves the processed JSONL file: `Digital_Music_5_sample.jsonl`.

### dataset.py

- Loads the JSONL dataset into a Hugging Face Dataset.
- Splits into train/test (10% test by default), but for demonstration and fine-tuning, all samples are used for training to maximize learning on a small dataset.
- Prints a sample entry for verification.

### finetune_qlora.py

- Loads the base LLM (e.g., `facebook/opt-1.3b`) with **4-bit quantization**.
- Loads tokenizer and sets `pad_token`.
- Tokenizes dataset into input-output sequences for SFT.
- Configures **QLoRA adapter** for parameter-efficient fine-tuning.
- Sets `TrainingArguments` (batch size, learning rate, epochs, fp16).
- Trains **only the QLoRA adapter**.
- Saves adapter weights and tokenizer to `outputs/qlora_opt_1.3b`

### generate_qlora.py

- Loads **base model** + **fine-tuned QLoRA adapter**.
- Loads a few samples from the dataset for comparison.
- Generates summaries **side by side** (base vs QLoRA).
- Optional: allows **custom user input** to summarize a review.
- Demonstrates the effect of QLoRA fine-tuning on instruction-following performance.


### chat_qlora.py

- Terminal-based interactive chatbot for summarizing reviews.
- Accepts user input continuously and outputs generated summaries.
- Demonstrates **multi-turn inference** with QLoRA.
- Maintains **conversation history** for context.

## Usage

**Preprocess dataset:**

python scripts/preprocess.py

**Train QLoRA adapter:**

python scripts/finetune_qlora.py --num_train_epochs 1 --per_device_train_batch_size 2 --learning_rate 2e-4 --fp16

**Compare base vs fine-tuned model outputs:**

python scripts/generate_qlora.py --num_samples 3 --max_new_tokens 64

**Run interactive chatbot:**

python scripts/chat_qlora.py --max_new_tokens 64


## model card
[detailed model card](./outputs/qlora_opt_1.3b/README.md)
