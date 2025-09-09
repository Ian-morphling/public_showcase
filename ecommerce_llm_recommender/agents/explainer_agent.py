import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional
from groq import Groq 

load_dotenv(override=True)

class ExplainerAgent:
    def __init__(self, groq_api_key: str = None):
        self.api_key = (
            groq_api_key
            or os.getenv("GROQ_API_KEY")
            or self._get_streamlit_secret("GROQ_API_KEY")
        )
        if not self.api_key:
            print("Warning: No GROQ API key provided. Using fallback responses.")
            self.client = None
        else:
            # initialize Groq client
            print(" ExplainerAgent initialized with key:", self.api_key[:8], "...")
            self.client = Groq(api_key=self.api_key)

    def build_rag_prompt(self, query: str, retrieved_docs: List[Dict], user_profile: Dict = None) -> str:
        """Construct a RAG-style prompt with retrieved reviews and user profile"""

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            distance = doc.get("faiss_distance", None)
            distance_str = f"{distance:.4f}" if distance is not None else "N/A"

            metadata = doc.get("metadata", {})
            rating = metadata.get("overall", "N/A")
            verified = metadata.get("verified", "N/A")

            context_parts.append(
                f"Review {i+1} (Similarity Score: {distance_str}):\n"
                f"Rating: {rating}/5 | Verified Purchase: {verified}\n"
                f"Content: {doc.get('text', '')}\n"
            )

        context = "\n".join(context_parts)

        user_context = ""
        if user_profile:
            if isinstance(user_profile, list) and len(user_profile) > 0:
                ratings = [r.get("rating", 0) for r in user_profile]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                total_reviews = len(user_profile)

                user_context = f"""
User Profile Context:
- Total Reviews: {total_reviews}
- Average Rating Given: {avg_rating:.1f}/5
- Recent Review Pattern: {ratings[-5:] if len(ratings) >= 5 else ratings}
"""
            elif isinstance(user_profile, dict):
                user_context = f"\nUser Profile: {json.dumps(user_profile, indent=2)}"

        prompt = f"""You are an expert e-commerce assistant helping users make informed purchasing decisions.

Based on the following product reviews and user context, provide a helpful, balanced, and detailed response to the user's question.

=== RETRIEVED REVIEWS ===
{context}

{user_context}

=== USER QUESTION ===
{query}

=== INSTRUCTIONS ===
- Analyze the reviews to extract key insights about the product
- Consider both positive and negative feedback
- If user profile is available, tailor recommendations to their preferences
- Provide specific examples from the reviews
- Be honest about potential drawbacks
- Give actionable advice for the purchase decision
- Keep response concise but informative (300-500 words)

Your response:"""

        return prompt

    def generate_answer(self, query: str, retrieved_docs: List[Dict], user_profile: Dict = None) -> str:
        """Generate answer using Groq API or fallback"""

        if not retrieved_docs:
            return "No relevant reviews found for your query. Please try a different search term."

        prompt = self.build_rag_prompt(query, retrieved_docs, user_profile)

        # Fallback if no API key
        if not self.client:
            return self._generate_fallback_response(query, retrieved_docs, user_profile)

        try:
            chat_completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful e-commerce assistant that analyzes product reviews."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
            )

            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            return f"Error calling Groq API: {str(e)}. Using fallback analysis..."

    def _generate_fallback_response(self, query: str, retrieved_docs: List[Dict], user_profile: Dict = None) -> str:
        """Generate a basic analysis when LLM API is not available"""

        if not retrieved_docs:
            return "No reviews found."

        ratings = []
        verified_count = 0
        total_docs = len(retrieved_docs)
        review_snippets = []

        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            rating = metadata.get("overall")
            if rating:
                ratings.append(rating)

            if metadata.get("verified"):
                verified_count += 1

            text = doc.get("text", "")
            if len(text) > 200:
                text = text[:200] + "..."
            review_snippets.append(f"• Rating {rating}/5: {text}")

        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        response = f"""**Analysis Summary for: "{query}"**

**Review Overview:**
- Found {total_docs} relevant reviews
- Average rating: {avg_rating:.1f}/5 stars
- Verified purchases: {verified_count}/{total_docs}

**Key Reviews:**
{chr(10).join(review_snippets[:3])}

**Quick Insights:**
"""
        if avg_rating >= 4:
            response += "- Generally positive feedback from users\n"
        elif avg_rating >= 3:
            response += "- Mixed reviews - consider pros and cons carefully\n"
        else:
            response += "- Several concerns raised in reviews\n"

        response += f"- {verified_count}/{total_docs} reviews from verified purchasers\n"

        if user_profile:
            response += f"\n**Personalized Note:** Based on your review history, this analysis considers your preferences."

        response += "\n\n*Note: LLM analysis unavailable - showing basic review summary*"

        return response
