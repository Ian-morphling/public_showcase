from datasets import load_dataset, Dataset
from pathlib import Path

DATA_PATH = Path("data/Digital_Music_5_sample.jsonl")  # output from preprocess.py

def load_sft_dataset(file_path=DATA_PATH):
    """
    Load the preprocessed JSONL into a Hugging Face Dataset.
    """
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # split into train/test
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    return dataset

def main():
    dataset = load_sft_dataset()
    print("Sample entry:", dataset['train'][0])

if __name__ == "__main__":
    main()
