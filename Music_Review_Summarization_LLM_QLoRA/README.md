---
base_model: facebook/opt-1.3b
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- text-generation
- music-review
---

# Music Review Summarization LLM (QLoRA)

Fine-tuned **OPT-1.3B** model using **QLoRA adapters** on Amazon Digital Music 5-core reviews. Generates concise, human-readable summaries of music reviews, demonstrating memory-efficient 4-bit fine-tuning suitable for small GPU environments.

## Model Details

### Model Description

This model is an OPT-1.3B transformer fine-tuned with QLoRA adapters for instruction-following summarization of Amazon Digital Music reviews. It produces short, readable summaries while being memory-efficient.

- **Model type:** Causal Language Model / Transformer  
- **Language(s):** English  
- **Finetuned from:** facebook/opt-1.3b  

### Training Data
Fine-tuned on a subset of 5,000 reviews from the Amazon Digital Music 5-core dataset.
This fine-tuning was done using QLoRA in 4-bit precision, enabling training on small GPUs while maintaining instruction-following performance.

## Uses

### Direct Use

- Summarizing Amazon Digital Music reviews in natural language.  
- Demonstrating instruction-following summarization for short-form text.  

## How to Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "facebook/opt-1.3b"
adapter_path = "outputs/qlora_opt_1.3b"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = PeftModel.from_pretrained(base_model, adapter_path)

prompt = "Summarize this review:\nThis album was amazing, but the last track was weak."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
