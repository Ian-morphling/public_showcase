import gzip, json, logging
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def review_is_valid(review: dict) -> bool:
    return (
        review.get("reviewText") and review.get("summary") and review.get("overall") is not None
        and review["reviewText"].strip() != ""
    )

def load_and_clean_reviews(file_path: str = "data/Electronics_5.json.gz",
                           output_dir: str = "outputs/clean_reviews/",
                           batch_size: int = 50_000) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = kept = part = 0

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            total += 1
            review = json.loads(line)
            if review_is_valid(review):
                rows.append(review)
                kept += 1

            if len(rows) >= batch_size:
                df = pd.DataFrame(rows)
                part_path = output_dir / f"part_{part:04d}.parquet"
                df.to_parquet(part_path, index=False, compression="snappy", engine="pyarrow")
                logging.info(f"Wrote {len(rows)} rows to {part_path}")
                rows.clear()
                part += 1

    if rows:
        df = pd.DataFrame(rows)
        part_path = output_dir / f"part_{part:04d}.parquet"
        df.to_parquet(part_path, index=False, compression="snappy", engine="pyarrow")
        logging.info(f"Wrote final {len(rows)} rows to {part_path}")

    logging.info(f"Processed {total} reviews. Kept {kept} clean reviews.")
    return part + 1  # number of parquet parts written


if __name__ == "__main__":
    load_and_clean_reviews()
