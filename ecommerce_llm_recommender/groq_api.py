import os
import requests

class GroqAPI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def call_llm(
        self, prompt: str, model: str = "llama3-70b-8192", temperature: float = 0.7, max_tokens: int = 512
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful e-commerce assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
