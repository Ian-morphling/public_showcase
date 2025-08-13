import pandas as pd
from pathlib import Path
import glob
from collections import defaultdict
import numpy as np
import math

# --- Config ---
DOWNSAMPLE_FRAC = 0.1            # 10% of data
MAX_ROWS_PER_FILE = 50_000       
OUTPUT_DIR = Path("outputs")
CLEAN_REVIEWS_DIR = OUTPUT_DIR / "clean_reviews"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_parts = sorted(glob.glob(str(CLEAN_REVIEWS_DIR / "part_*.parquet")))

    user_profiles = defaultdict(list)
    doc_batch = []
    profile_batch = []
    doc_part = 0
    profile_part = 0

    print(f"Processing {len(cleaned_parts)} cleaned parquet parts...")

    for part_file in cleaned_parts:
        df = pd.read_parquet(part_file)

        # --- Downsample 10% ---
        df = df.sample(frac=DOWNSAMPLE_FRAC, random_state=42)

        for _, row in df.iterrows():
            review_text = row['reviewText'].strip()
            summary_text = row['summary'].strip()
            full_text = f"{review_text}\n\nSummary: {summary_text}"

            rating = row['overall']
            review_length = len(review_text.split())
            verified = row.get('verified', False)
            reviewer_id = row['reviewerID']
            reviewer_name = row.get('reviewerName', '')
            review_time = row.get('reviewTime')
            asin = row['asin']

            # --- Rating label ---
            if rating >= 4:
                rating_label = "positive"
            elif rating == 3:
                rating_label = "neutral"
            else:
                rating_label = "negative"

            # --- Helpful votes ---
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

            # --- Flattened document row ---
            doc_row = {
                "id": f"{asin}_{reviewer_id}",
                "text": full_text,
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
            doc_batch.append(doc_row)

            # --- Flattened user profile row ---
            profile_row = {
                "reviewerID": reviewer_id,
                "asin": asin,
                "rating": rating,
                "rating_label": rating_label,
                "review_length": review_length,
                "summary": summary_text,
                "review": review_text,
                "has_votes": has_votes,
                "votes": votes,
                "vote_bin": vote_bin
            }
            profile_batch.append(profile_row)

            # --- Write batch if max rows reached ---
            if len(doc_batch) >= MAX_ROWS_PER_FILE:
                df_docs = pd.DataFrame(doc_batch)
                df_docs.to_parquet(
                    OUTPUT_DIR / f"documents_part_{doc_part:04d}.parquet",
                    index=False,
                    compression="snappy",
                    engine="pyarrow"
                )
                print(f"Wrote documents_part_{doc_part:04d}.parquet with {len(doc_batch)} rows")
                doc_batch.clear()
                doc_part += 1

            if len(profile_batch) >= MAX_ROWS_PER_FILE:
                df_profiles = pd.DataFrame(profile_batch)
                df_profiles.to_parquet(
                    OUTPUT_DIR / f"user_profiles_part_{profile_part:04d}.parquet",
                    index=False,
                    compression="snappy",
                    engine="pyarrow"
                )
                print(f"Wrote user_profiles_part_{profile_part:04d}.parquet with {len(profile_batch)} rows")
                profile_batch.clear()
                profile_part += 1

    # --- Flush remaining docs ---
    if doc_batch:
        df_docs = pd.DataFrame(doc_batch)
        df_docs.to_parquet(
            OUTPUT_DIR / f"documents_part_{doc_part:04d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow"
        )
        print(f"Wrote documents_part_{doc_part:04d}.parquet with {len(doc_batch)} rows")

    if profile_batch:
        df_profiles = pd.DataFrame(profile_batch)
        df_profiles.to_parquet(
            OUTPUT_DIR / f"user_profiles_part_{profile_part:04d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow"
        )
        print(f"Wrote user_profiles_part_{profile_part:04d}.parquet with {len(profile_batch)} rows")

    print("Finished processing documents and user profiles.")

if __name__ == "__main__":
    main()