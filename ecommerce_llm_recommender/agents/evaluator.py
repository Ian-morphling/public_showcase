import json
import pandas as pd

class Evaluator:
    def __init__(self, retriever, ground_truth_json, k=5):
        self.retriever = retriever
        self.k = k
        with open(ground_truth_json, "r", encoding="utf-8") as f:
            self.ground_truth = json.load(f)

    def evaluate_query(self, query):
        """
        Evaluate a single query: compute precision, recall, F1
        """
        top_docs = self.retriever.retrieve(query, top_k=self.k)
        predicted_ids = [d["asin"] for d in top_docs if "asin" in d]

        true_ids = set(self.ground_truth.get(query, []))
        predicted_ids_set = set(predicted_ids)

        tp = len(predicted_ids_set & true_ids)
        fp = len(predicted_ids_set - true_ids)
        fn = len(true_ids - predicted_ids_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Return results + top docs summary
        summary_docs = [
            {
                "asin": d.get("asin"),
                "title": d.get("title"),
                "score": d.get("score", None)
            } for d in top_docs
        ]

        return {
            "query": query,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved_docs": summary_docs
        }

    def evaluate_all(self, output_csv):
        results = []
        for query in self.ground_truth:
            res = self.evaluate_query(query)
            results.append(res)

            # --- Print per query in CLI ---
            print(f"\nQuery: {res['query']}")
            print(f"Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | F1: {res['f1']:.4f}")
            print("Top retrieved docs:")
            for doc in res["retrieved_docs"]:
                print(f"  ASIN: {doc['asin']} | Title: {doc['title']} | Score: {doc['score']}")

        # Flatten for CSV
        rows = []
        for r in results:
            for doc in r["retrieved_docs"]:
                rows.append({
                    "query": r["query"],
                    "asin": doc["asin"],
                    "title": doc["title"],
                    "score": doc["score"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"]
                })

        df = pd.DataFrame(rows)

        # Append summary row per metric
        summary = {
            "query": "SUMMARY",
            "asin": "",
            "title": "",
            "score": "",
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "f1": df["f1"].mean()
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        df.to_csv(output_csv, index=False)
        print(f"\nEvaluation completed. Results saved to {output_csv}")
        print(f"Summary - Precision: {summary['precision']:.4f} | Recall: {summary['recall']:.4f} | F1: {summary['f1']:.4f}")
