import pandas as pd
import json
from pathlib import Path
import glob
from collections import defaultdict
import numpy as np

BATCH_SIZE = 100_000
OUTPUT_DIR = Path("outputs")
CLEAN_REVIEWS_DIR = OUTPUT_DIR / "clean_reviews"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_parts = sorted(glob.glob(str(CLEAN_REVIEWS_DIR / "part_*.parquet")))

    user_profiles = defaultdict(list)       # dictionary of reviewerID to list of reviews.
    doc_part = 0
    doc_batch = []

    for part_file in cleaned_parts:
        df = pd.read_parquet(part_file)

        for _, row in df.iterrows():
            review_text = row['reviewText'].strip()
            summary_text = row['summary'].strip()
            full_text = review_text + "\n\nSummary: " + summary_text

            # --------- Basic Metadata ---------
            rating = row['overall']
            review_length = len(review_text.split())
            verified = row.get('verified', False)
            reviewer_id = row['reviewerID']
            reviewer_name = row.get('reviewerName', '')
            review_time = row.get('reviewTime')
            asin = row['asin']

            # --------- Rating Label (for personalization/recommendation) ---------
            if rating >= 4:
                rating_label = "positive"
            elif rating == 3:
                rating_label = "neutral"
            else:
                rating_label = "negative"

            # --------- Helpful Votes Handling ---------
            vote_raw = row.get("vote", None)
            has_votes = vote_raw is not None
            try:
                votes = int(vote_raw) if has_votes else None
            except ValueError:
                votes = None
                has_votes = False

            if votes is not None:
                if votes >= 20:
                    vote_bin = "high"
                elif votes >= 5:
                    vote_bin = "medium"
                elif votes >= 1:
                    vote_bin = "low"
                else:
                    vote_bin = "none"
            else:
                vote_bin = "missing"

            # --------- Metadata Package ---------
            metadata = {
                "asin": asin,
                "overall": rating,
                "rating_label": rating_label,
                "review_length": review_length,
                "verified": verified,
                "reviewTime": review_time,
                "reviewerID": reviewer_id,
                "reviewerName": reviewer_name,
                "has_votes": has_votes,
                "votes": votes,
                "vote_bin": vote_bin
            }

            # --------- Document for RAG ---------
            doc = {
                "id": f"{asin}_{reviewer_id}",
                "text": full_text,
                "metadata": metadata
            }
            doc_batch.append(doc)

            # --------- User profile storage ---------
            user_profiles[reviewer_id].append({
                "asin": asin,
                "rating": rating,
                "rating_label": rating_label,
                "review_length": review_length,
                "summary": summary_text,
                "review": review_text,
                "has_votes": has_votes,
                "votes": votes,
                "vote_bin": vote_bin
            })

            # Write batch to file
            if len(doc_batch) >= BATCH_SIZE:
                df_docs = pd.DataFrame(doc_batch)
                df_docs.to_parquet(OUTPUT_DIR / f"documents_part_{doc_part:04d}.parquet", index=False)
                print(f"Wrote documents_part_{doc_part:04d}.parquet with {len(doc_batch)} records")
                doc_batch.clear()
                doc_part += 1

    # Final flush
    if doc_batch:
        df_docs = pd.DataFrame(doc_batch)
        df_docs.to_parquet(OUTPUT_DIR / f"documents_part_{doc_part:04d}.parquet", index=False)
        print(f"Wrote documents_part_{doc_part:04d}.parquet with {len(doc_batch)} records")

    # Save user profiles JSON
    with open(OUTPUT_DIR / "user_profiles.json", "w") as f:
        json.dump(user_profiles, f, indent=2)
    print("Saved user profiles to outputs/user_profiles.json")


if __name__ == "__main__":
    main()
