import os
from typing import List, Dict, Optional
from groq import Groq
import json

class JudgeAgent:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: No GROQ API key provided. Using fallback.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

    def build_judge_prompt(
        self, query: str, explanation: str, retrieved_docs: List[Dict]
    ) -> str:
        context = "\n".join(
            [
                f"Review {i+1} (Rating {doc.get('overall')}, Verified {doc.get('verified')}): {doc.get('text','')}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

        prompt = f"""
You are an expert evaluator for AI product recommendations.

User Query:
{query}

Retrieved Reviews:
{context}

AI Explanation:
{explanation}

Please score the AI explanation on a scale of 0–5 for each criterion:
1. Relevance: Does the explanation answer the user query?
2. Groundedness: Is it faithful to the retrieved reviews without unsupported claims?
3. Balance: Does it fairly present pros and cons from the reviews?

Return a JSON object like:
{{
"Relevance": 4,
"Groundedness": 5,
"Balance": 3,
"Justification": "Brief reasoning for each score"
}}
"""
        return prompt

    def judge(
        self, query: str, explanation: str, retrieved_docs: List[Dict]
    ) -> Dict:
        # Always return a dict with all keys
        default_output = {
            "Relevance": None,
            "Groundedness": None,
            "Balance": None,
            "Justification": ""
        }

        if not self.client:
            return {**default_output, "Justification": "Groq API not available"}

        prompt = self.build_judge_prompt(query, explanation, retrieved_docs)

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an AI evaluator for product recommendations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()

            # parse JSON
            try:
                parsed = json.loads(content)
                # Handle nested JSON if Justification is stringified JSON
                if isinstance(parsed.get("Justification"), str):
                    try:
                        nested = json.loads(parsed["Justification"])
                        for key in ["Relevance", "Groundedness", "Balance"]:
                            if key in nested:
                                parsed[key] = nested[key]
                        parsed["Justification"] = nested.get("Justification", parsed["Justification"])
                    except Exception:
                        # Keep original Justification string if nested parse fails
                        pass
                # Ensure all keys exist
                for key in default_output:
                    if key not in parsed:
                        parsed[key] = default_output[key]
                return parsed
            except Exception:
                # fallback: return raw content as Justification
                return {**default_output, "Justification": content}

        except Exception as e:
            return {**default_output, "Justification": str(e)}
