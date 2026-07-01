from dotenv import load_dotenv
import time
from google import genai
from google.genai import types
from google.genai import errors

# Load API key from .env file
load_dotenv()

client = genai.Client(
    # If Gemini does not respond in 60 seconds, raise a timeout error
    http_options=types.HttpOptions(timeout=60000)
)

def call_gemini(prompt, model="gemini-3.5-flash", max_retries=3):
    """Call Gemini once with the given prompt and model, retrying up to max_retries times if necessary."""
    # For each attempt, try to stream the response from Gemini.
    for attempt in range(1, max_retries + 1):
        partial_text = ""
        usage = None
        finish_reason = None
        start = time.time()
        try:
            # Stream the response from Gemini, which allows us to handle partial responses and errors more gracefully.
            print(f"[{model}] [Attempt {attempt}] Streaming request...")
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
            ):
                # If the chunk contains text, append it to the partial_text variable.
                if chunk.text:
                    partial_text += chunk.text
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
                if chunk.candidates:
                    finish_reason = chunk.candidates[0].finish_reason
            # The elapsed time is calculated after the streaming is complete
            elapsed = time.time() - start
            print(f"[{model}] [Attempt {attempt}] Stream finished after {elapsed:.1f}s")
            # If no text was returned, it likely means the request was blocked or failed, so we skip this attempt.
            if not partial_text:
                print(f"[{model}] No text returned (likely blocked). Skipping.")
                return None
            # Return the result as a dictionary
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
        
        # Handle ServerError exceptions
        except errors.ServerError as e:
            elapsed = time.time() - start
            if partial_text:
                print(
                    # Log a warning that we got a server error but still received some text before it failed.
                    f"[{model}] [Attempt {attempt}] Server error after {elapsed:.1f}s, "
                    f"but got {len(partial_text)} chars before it died. Returning partial result."
                )
                # Return the partial result with a flag indicating that it is partial.
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
            # If no text was returned, we will retry after an exponential backoff.
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Server error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

        # Handle ClientError exceptions, which are usually not worth retrying unless they are rate limit errors
        except errors.ClientError as e:
            # 4xx errors (400, 401, 403, 429) — usually NOT worth blind retrying
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30
                print(f"[{model}] [Attempt {attempt}] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[{model}] Client error (not retryable): {e}")
                return None
        # Handle any other unexpected exceptions that may occur during the API call
        except Exception as e:
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Unexpected error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    # If we reach this point, it means all retries have been exhausted without a successful response, so we log that and return None.
    print(f"[{model}] Max retries exceeded. Giving up on this model.")
    return None
