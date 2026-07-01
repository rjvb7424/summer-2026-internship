from dotenv import load_dotenv
load_dotenv()

import time
from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client(
    # If Gemini does not respond in 60 seconds, raise a timeout error
    http_options=types.HttpOptions(timeout=60000)
)

def call_gemini(prompt, model="gemini-3.5-flash", max_retries=3):
    """Call the Gemini API with the given prompt and model, handling retries for transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Attempt {attempt}] Sending request...")
            start = time.time()

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=1024)
                ),
            )
            elapsed = time.time() - start
            print(f"[Attempt {attempt}] Got response after {elapsed:.1f}s")

            if not response.candidates:
                print("No candidates returned (likely blocked). Skipping.")
                return None

            candidate = response.candidates[0]

            if candidate.finish_reason != "STOP":
                print(f"Non-normal finish reason: {candidate.finish_reason}")

            usage = response.usage_metadata

            return {
                "text": response.text,
                "elapsed_seconds": elapsed,
                "prompt_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "thinking_tokens": getattr(usage, "thoughts_token_count", None),
                "total_tokens": usage.total_token_count,
                "finish_reason": candidate.finish_reason,
                "model_version": response.model_version,
            }

        except errors.ServerError as e:
            # 5xx errors (500, 503, 504) — transient, worth retrying
            wait = 2 ** attempt
            print(f"[Attempt {attempt}] Server error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

        except errors.ClientError as e:
            # 4xx errors (400, 401, 403, 429) — usually NOT worth blind retrying
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30
                print(f"[Attempt {attempt}] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Client error (not retryable): {e}")
                return None

        except Exception as e:
            # Catch-all: network errors, SSL issues, etc.
            wait = 2 ** attempt
            print(f"[Attempt {attempt}] Unexpected error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print("Max retries exceeded. Giving up on this call.")
    return None


if __name__ == "__main__":
    result = call_gemini("How would you solve a maze?")
    if result:
        print("\n--- Response ---")
        print(result["text"])
        print("\n--- Metadata ---")
        print(f"Model version: {result['model_version']}")
        print(f"Time: {result['elapsed_seconds']:.2f}s")
        print(f"Tokens — prompt: {result['prompt_tokens']}, output: {result['output_tokens']}, "
              f"thinking: {result['thinking_tokens']}, total: {result['total_tokens']}")
        print(f"Finish reason: {result['finish_reason']}")