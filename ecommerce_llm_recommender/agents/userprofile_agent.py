from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

class UserProfileAgent:
    def __init__(self, user_profiles_dir: str):
        self.user_profiles_dir = Path(user_profiles_dir)
        if not self.user_profiles_dir.exists():
            raise FileNotFoundError(f"User profiles directory not found: {user_profiles_dir}")

        # Load all Parquet files
        self.parquet_files = sorted(self.user_profiles_dir.glob("user_profiles_part_*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"No user profile Parquet files found in {user_profiles_dir}")

        # Index reviewer IDs for quick lookup
        self._profiles_index = None
        self._load_profiles_index()

    def _load_profiles_index(self):
        if self._profiles_index is None:
            self._profiles_index = set()
            for f in self.parquet_files:
                df = pd.read_parquet(f, columns=["reviewerID"])
                self._profiles_index.update(df["reviewerID"].unique())

    def has_user(self, reviewer_id: str) -> bool:
        return reviewer_id in self._profiles_index

    def get_user_stats(self, reviewer_id: str, max_reviews: int = 50) -> Optional[List[Dict]]:
        if not self.has_user(reviewer_id):
            return None

        user_rows = []
        for f in self.parquet_files:
            df = pd.read_parquet(f)
            user_df = df[df["reviewerID"] == reviewer_id]
            if not user_df.empty:
                user_rows.extend(user_df.to_dict(orient="records"))
                if len(user_rows) >= max_reviews:
                    break

        return user_rows[-max_reviews:] if user_rows else None

    def get_user_summary(self, reviewer_id: str) -> Optional[Dict]:
        user_data = self.get_user_stats(reviewer_id, max_reviews=100)
        if not user_data:
            return None

        ratings = [r.get('rating', 0) for r in user_data]
        return {
            'total_reviews': len(user_data),
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0,
            'rating_distribution': {
                '5_star': len([r for r in ratings if r == 5]),
                '4_star': len([r for r in ratings if r == 4]),
                '3_star': len([r for r in ratings if r == 3]),
                '2_star': len([r for r in ratings if r == 2]),
                '1_star': len([r for r in ratings if r == 1]),
            },
            'verified_purchases': len([r for r in user_data if r.get('verified', False)]),
        }