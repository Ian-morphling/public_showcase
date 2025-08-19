# Usage examples: 
# python -m scripts.run_evaluation --query "best wireless headphones under $200" 
# python -m scripts.run_evaluation --queries_file scripts/queries.txt

import argparse
from pathlib import Path
import requests

from scripts.generate_ground_truth import generate_ground_truth_from_queries
from agents.retriever_agent import RetrieverAgent
from agents.evaluator import Evaluator

def download_file(url, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        print(f"Downloading {target_path.name}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {target_path.name}")
    else:
        print(f"{target_path.name} already exists")
    return target_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="index/product.index")
    parser.add_argument("--mapping_path", default="index/id_to_filename.pkl")
    parser.add_argument("--docs_dir", default="outputs")
    parser.add_argument("--query", default=None, help="Single query to evaluate")
    parser.add_argument("--queries_file", default=None, help="File containing queries, one per line")
    parser.add_argument("--ground_truth_json", default="data/ground_truth.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_url", default=None)
    parser.add_argument("--mapping_url", default=None)
    parser.add_argument("--output_csv", default="outputs/evaluation_results.csv")
    args = parser.parse_args()

    if args.index_url:
        download_file(args.index_url, args.index_path)
    if args.mapping_url:
        download_file(args.mapping_url, args.mapping_path)

    retriever = RetrieverAgent(
        index_path=args.index_path,
        mapping_path=args.mapping_path,
        docs_dir=args.docs_dir
    )

    if args.queries_file:
        # generate ground truth automatically
        generate_ground_truth_from_queries(args.queries_file, args.docs_dir, args.ground_truth_json)
    elif args.query:
        # save single query to temp file for ground truth generation
        temp_queries_file = Path("temp_single_query.txt")
        temp_queries_file.write_text(args.query)
        generate_ground_truth_from_queries(temp_queries_file, args.docs_dir, args.ground_truth_json)
        temp_queries_file.unlink()
    else:
        raise ValueError("Provide either --query or --queries_file")

    evaluator = Evaluator(retriever, args.ground_truth_json, k=args.top_k)
    evaluator.evaluate_all(args.output_csv)

if __name__ == "__main__":
    main()
