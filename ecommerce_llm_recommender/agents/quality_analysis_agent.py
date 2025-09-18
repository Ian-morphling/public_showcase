from typing import List, Dict

class QualityAnalysisAgent:
    """
    Computes a quality score for each review based on key metadata fields:
    - overall rating (1-5)
    - votes (helpfulness)
    - verified purchase
    - vote_bin ("high", "medium", "low", "none")
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_quality_score(doc: Dict, max_votes: int = 50) -> Dict:
        """
        Compute a numeric quality score and label for a single review.
        Returns the doc with added keys: "quality_score" and "quality_label".
        """

        # --- Extract metadata ---
        overall = doc.get("overall", 0)
        votes = doc.get("votes", 0)
        verified = doc.get("verified", False)
        vote_bin = doc.get("vote_bin", "none")

        # --- Normalize factors ---
        rating_score = min(max(float(overall) / 5.0, 0), 1)
        votes_score = min(int(votes) / max_votes, 1.0) if votes else 0
        verified_score = 1.0 if verified else 0.0

        # Map vote_bin to numeric score
        vote_bin_mapping = {
            "high": 1.0,
            "medium": 0.66,
            "low": 0.33,
            "none": 0.0,
            "missing": 0.0
        }
        vote_bin_score = vote_bin_mapping.get(vote_bin, 0.0)

        # --- Weighted sum for overall quality ---
        quality_numeric = (
            0.5 * rating_score +
            0.2 * votes_score +
            0.2 * verified_score +
            0.1 * vote_bin_score
        )

        # --- Assign label ---
        if quality_numeric >= 0.75:
            quality_label = "High"
        elif quality_numeric >= 0.5:
            quality_label = "Medium"
        else:
            quality_label = "Low"

        # --- Update doc ---
        doc["quality_score"] = round(quality_numeric, 3)
        doc["quality_label"] = quality_label

        return doc

    def analyze_reviews(self, docs: List[Dict]) -> List[Dict]:
        """
        Apply quality scoring to a list of review documents.
        """
        return [self.compute_quality_score(d) for d in docs]
