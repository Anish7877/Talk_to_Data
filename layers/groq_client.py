"""
Groq API Client
FIX: Updated to current active Groq models (llama3-8b-8192 was decommissioned)
"""

from typing import Dict, Any, List, Optional
import os
import requests


class GroqClient:
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com\n"
                "Then run: export GROQ_API_KEY=your_key_here"
            )
        self.default_model = model
        self.base_url = self.BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completions_create(
        self,
        model: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Dict[str, str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import time

        if messages is None:
            messages = []
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            payload["response_format"] = response_format
        payload.update(kwargs)

        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=60
            )

            if response.status_code == 429:
                wait = 2 ** attempt
                try:
                    msg = response.json()["error"]["message"]
                    if "try again in" in msg:
                        wait = float(msg.split("try again in")[1].split("s")[0].strip()) + 0.5
                except Exception:
                    pass
                print(f"[GROQ] Rate limit hit — waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise Exception(f"Groq API error {response.status_code}: {response.text}")

            return response.json()

        raise Exception(f"Rate limit exceeded after {MAX_RETRIES} retries. Wait a moment and try again.")

    def create(self, model=None, messages=None, temperature=0.0, max_tokens=1024, **kwargs):
        return self.chat_completions_create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, **kwargs
        )


_instance = None

def get_groq_client() -> GroqClient:
    global _instance
    if _instance is None:
        _instance = GroqClient()
    return _instance

def reset_groq_client():
    global _instance
    _instance = None


# FIX: Updated to current active Groq models (verified April 2025)
# llama3-8b-8192 and llama3-70b-8192 were decommissioned
GROQ_MODELS = {
    "fast":     "llama-3.1-8b-instant",    # Fast, low latency
    "medium":   "llama-3.3-70b-versatile", # Balanced
    "powerful": "llama-3.3-70b-versatile", # Most capable
}


if __name__ == "__main__":
    try:
        client = GroqClient()
        print("Groq client initialized!")
        print(f"Models: {GROQ_MODELS}")
    except ValueError as e:
        print(f"Error: {e}")
