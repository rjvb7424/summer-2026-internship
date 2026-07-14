"""Gemini agent: Crafter policy backed by the Google Gemini API.

Requires the GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.
"""

import time

import config
from base_agent import SYSTEM_PROMPT, BaseAgent


class GeminiAgent(BaseAgent):
    """Wraps one Gemini model as a Crafter policy."""

    provider = "gemini"

    def __init__(self, model_name):
        super().__init__(model_name)
        from google import genai  # lazy: only needed when a gemini model runs
        from google.genai import types
        self.types = types
        self.client = genai.Client(
            http_options=types.HttpOptions(timeout=config.API_TIMEOUT_SEC * 1000)
        )

    def _generate(self, user_text):
        generation_config = self.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.API_TEMPERATURE,
            max_output_tokens=config.API_MAX_TOKENS,
        )
        last_error = None
        for attempt in range(config.API_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_text,
                    config=generation_config,
                )
                return (response.text or "").strip()
            except Exception as error:
                last_error = error
                time.sleep(config.API_RETRY_DELAY_SEC * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after retries: {last_error}")
