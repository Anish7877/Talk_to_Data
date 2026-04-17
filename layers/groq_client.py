"""
Groq API Client
Updated with Mock Mode and Ollama Local Support
"""

from typing import Dict, Any, List, Optional
import os
import requests

class GroqClient:
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant"):
        # Check for testing modes
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        self.mock_mode = os.getenv("MOCK_LLM_MODE", "false").lower() == "true"

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        # Bypass API key check if we are doing local testing
        if not self.api_key and not (self.use_ollama or self.mock_mode):
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com\n"
                "Then run: export GROQ_API_KEY=your_key_here"
            )

        self.default_model = model

        # Route traffic to localhost if using Ollama
        if self.use_ollama:
            self.base_url = "http://localhost:11434/v1"
        else:
            self.base_url = self.BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        # Ollama doesn't need a real API key, but it ignores this safely
        return {
            "Authorization": f"Bearer {self.api_key or 'mock_key'}",
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

        # --- Mock Mode Intercept ---
        if getattr(self, 'mock_mode', False):
            prompt_str = str(messages).lower()

            # 1. Mock Intent Router (Must be valid JSON)
            if "json" in prompt_str and "route" in prompt_str:
                dummy_response = '{"route": "rag", "schemas": [], "confidence": 0.99, "reasoning": "Mocked route"}'

            # 2. Mock SQL Generator
            elif "sql" in prompt_str or "select" in prompt_str:
                dummy_response = "SELECT * FROM mock_table LIMIT 5;"

            # 3. Mock Storyteller
            else:
                dummy_response = "This is a free mocked response! Your pipeline successfully reached the end."

            return {"choices": [{"message": {"content": dummy_response}}]}

        # --- Ollama / Standard Routing ---
        model = model or self.default_model
        if getattr(self, 'use_ollama', False):
            # Groq model names don't exist in Ollama. Map everything to standard llama3.1
            model = "llama3.1"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Ollama sometimes rejects Groq's exact json object format flag, so we skip it for local
        if response_format and not self.use_ollama:
            payload["response_format"] = response_format

        payload.update(kwargs)

        # --- The Network Request Loop ---
        MAX_RETRIES = 5
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60
                )
            except requests.exceptions.ConnectionError:
                if self.use_ollama:
                    raise Exception("Ollama is not running. Please start it with 'ollama run llama3.1'")
                raise

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
                raise Exception(f"API error {response.status_code}: {response.text}")

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

GROQ_MODELS = {
    "fast":     "llama-3.1-8b-instant",
    "medium":   "llama-3.3-70b-versatile",
    "powerful": "llama-3.3-70b-versatile",
}
