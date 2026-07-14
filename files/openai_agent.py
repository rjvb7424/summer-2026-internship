"""OpenAI agent: Crafter policy backed by the ChatGPT API.

Requires the OPENAI_API_KEY environment variable.
"""

import time

import config
from base_agent import SYSTEM_PROMPT, BaseAgent


class OpenAIAgent(BaseAgent):
    """Wraps one OpenAI chat model as a Crafter policy."""

    provider = "openai"

    def __init__(self, model_name):
        super().__init__(model_name)
        from openai import OpenAI  # lazy: only needed when an openai model runs
        self.client = OpenAI(timeout=config.API_TIMEOUT_SEC, max_retries=0)
        self.supports_temperature = True  # o-series/reasoning models reject it

    def _generate(self, user_text):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        last_error = None
        for attempt in range(config.API_MAX_RETRIES):
            try:
                params = dict(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=config.API_MAX_TOKENS,
                )
                if self.supports_temperature:
                    params["temperature"] = config.API_TEMPERATURE
                response = self.client.chat.completions.create(**params)
                return (response.choices[0].message.content or "").strip()
            except Exception as error:
                if self.supports_temperature and "temperature" in str(error):
                    self.supports_temperature = False
                    continue  # retry immediately without the parameter
                last_error = error
                time.sleep(config.API_RETRY_DELAY_SEC * (attempt + 1))
        raise RuntimeError(f"OpenAI call failed after retries: {last_error}")
