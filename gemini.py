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


def _call_gemini_once(prompt, model, max_retries=3, thinking_budget=512):
    """Try a single model with retries, streaming the response.

    Streaming means that if the server errors out partway through (deadline
    exceeded, overload, etc.), whatever text already arrived is kept and
    returned instead of being thrown away — unlike a plain (non-streaming)
    call, which only gives you anything at all once the ENTIRE response has
    finished generating server-side.
    """
    for attempt in range(1, max_retries + 1):
        partial_text = ""
        usage = None
        finish_reason = None
        start = time.time()
        try:
            print(f"[{model}] [Attempt {attempt}] Streaming request...")
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
                ),
            ):
                if chunk.text:
                    partial_text += chunk.text
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
                if chunk.candidates:
                    finish_reason = chunk.candidates[0].finish_reason

            elapsed = time.time() - start
            print(f"[{model}] [Attempt {attempt}] Stream finished after {elapsed:.1f}s")

            if not partial_text:
                print(f"[{model}] No text returned (likely blocked). Skipping.")
                return None

            return {
                "text": partial_text,
                "elapsed_seconds": elapsed,
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "thinking_tokens": getattr(usage, "thoughts_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
                "finish_reason": finish_reason,
                "model_version": model,
                "is_partial": False,
            }

        except errors.ServerError as e:
            elapsed = time.time() - start
            # 5xx errors (500, 503, 504) — transient. But if the stream had
            # already produced text before dying, keep it rather than
            # discarding a real (if incomplete) answer.
            if partial_text:
                print(
                    f"[{model}] [Attempt {attempt}] Server error after {elapsed:.1f}s, "
                    f"but got {len(partial_text)} chars before it died. Returning partial result."
                )
                return {
                    "text": partial_text,
                    "elapsed_seconds": elapsed,
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                    "thinking_tokens": getattr(usage, "thoughts_token_count", None),
                    "total_tokens": getattr(usage, "total_token_count", None),
                    "finish_reason": finish_reason,
                    "model_version": model,
                    "is_partial": True,
                }
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Server error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

        except errors.ClientError as e:
            # 4xx errors (400, 401, 403, 429) — usually NOT worth blind retrying
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30
                print(f"[{model}] [Attempt {attempt}] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[{model}] Client error (not retryable): {e}")
                return None

        except Exception as e:
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Unexpected error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"[{model}] Max retries exceeded. Giving up on this model.")
    return None


def call_gemini(prompt, models=("gemini-3.5-flash", "gemini-2.5-flash"), max_retries=3):
    """Call Gemini, falling through a chain of models if the earlier ones
    fail after their own retries with zero usable text."""
    if isinstance(models, str):
        models = (models,)

    for model in models:
        result = _call_gemini_once(prompt, model, max_retries=max_retries)
        if result is not None:
            return result
        print(f"[{model}] exhausted, falling back to next model in chain...")

    print("All models in the fallback chain failed. Giving up on this call.")
    return None


if __name__ == "__main__":
    result = call_gemini("How would you solve a maze?")
    if result:
        print("\n--- Response ---")
        print(result["text"])
        print("\n--- Metadata ---")
        print(f"Model version: {result['model_version']}")
        print(f"Time: {result['elapsed_seconds']:.2f}s")
        print(f"Partial: {result['is_partial']}")
        print(f"Tokens — prompt: {result['prompt_tokens']}, output: {result['output_tokens']}, "
              f"thinking: {result['thinking_tokens']}, total: {result['total_tokens']}")
        print(f"Finish reason: {result['finish_reason']}")