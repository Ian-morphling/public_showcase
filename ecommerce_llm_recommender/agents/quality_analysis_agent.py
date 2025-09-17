from typing import List, Dict

class QualityAnalysisAgent:
    """
    Computes a quality score for each review based on metadata:
    - overall rating
    - votes (helpfulness)
    - verified purchase
    - vote_bin (highly helpful or not)
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_quality_score(doc: Dict, max_votes: int = 50) -> Dict:
        metadata = doc.get("metadata", {})

        overall = metadata.get("overall", 0)
        votes = metadata.get("votes", 0)
        verified = metadata.get("verified", False)
        vote_bin = metadata.get("vote_bin", 0)

        # Normalize each factor to 0-1
        rating_score = min(max(overall / 5.0, 0), 1)
        votes_score = min(votes / max_votes, 1.0)  # smoother scaling
        verified_score = 1.0 if verified else 0.0
        vote_bin_score = 1.0 if vote_bin else 0.0

        # Weighted sum
        quality_numeric = (0.5 * rating_score +
                           0.2 * votes_score +
                           0.2 * verified_score +
                           0.1 * vote_bin_score)

        # Low / Medium / High
        if quality_numeric >= 0.75:
            quality_label = "High"
        elif quality_numeric >= 0.5:
            quality_label = "Medium"
        else:
            quality_label = "Low"

        doc["quality_score"] = round(quality_numeric, 3)
        doc["quality_label"] = quality_label
        return doc

    def analyze_reviews(self, docs: List[Dict]) -> List[Dict]:
        return [self.compute_quality_score(d) for d in docs]