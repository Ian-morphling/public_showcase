import json
import gzip
import random
from pathlib import Path

DATA_PATH = Path("data/Digital_Music_5.json.gz") 
OUTPUT_PATH = Path("data/Digital_Music_5_sample.jsonl")
SAMPLE_SIZE = 5000 

def load_reviews(file_path):
    """Load reviews from gzipped JSON file."""
    reviews = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    return reviews

def preprocess_reviews(reviews):
    """
    Convert reviews to a simplified dict for SFT:
    - instruction: task instruction for model
    - input: original review text
    - output: summary
    """
    processed = []
    for review in reviews:
        if "reviewText" in review and "summary" in review:
            processed.append({
                "instruction": "Summarize this review.",
                "input": review["reviewText"],
                "output": review["summary"]
            })
    return processed

def main():
    print("Loading reviews...")
    reviews = load_reviews(DATA_PATH)
    print(f"Total reviews loaded: {len(reviews)}")
    
    print(f"Sampling {SAMPLE_SIZE} reviews...")
    sampled = random.sample(reviews, min(SAMPLE_SIZE, len(reviews)))
    
    print("Preprocessing reviews...")
    processed = preprocess_reviews(sampled)
    
    print(f"Saving preprocessed data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")
    print("Done!")

if __name__ == "__main__":
    main()