from dotenv import load_dotenv
import time
from openai import OpenAI
from openai import (APIConnectionError, APIStatusError, InternalServerError, RateLimitError)

# Load API key from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

def call_gpt(prompt, model="gpt-5.2", max_retries=3):
    """Call ChatGPT once with the given prompt and model, retrying up to max_retries times if necessary."""
    # For each attempt, try to stream the response from ChatGPT.
    for attempt in range(1, max_retries + 1):
        partial_text = ""
        usage = None
        finish_reason = None
        start = time.time()
        try:
            # Stream the response from ChatGPT, which allows us to handle partial responses and errors more gracefully.
            print(f"[{model}] [Attempt {attempt}] Streaming request...")
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                # If the chunk contains text, append it to the partial_text variable.
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        partial_text += delta.content
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
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
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "thinking_tokens": getattr(
                    getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None
                ),
                "total_tokens": getattr(usage, "total_tokens", None),
                "finish_reason": finish_reason,
                "model_version": model,
                "is_partial": False,
            }

        # Handle InternalServerError exceptions
        except InternalServerError as e:
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
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "thinking_tokens": getattr(
                        getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None
                    ),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "finish_reason": finish_reason,
                    "model_version": model,
                    "is_partial": True,
                }
            # If no text was returned, we will retry after an exponential backoff.
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Server error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

        # Handle RateLimitError exceptions.
        except RateLimitError as e:
            wait = 30
            print(f"[{model}] [Attempt {attempt}] Rate limited. Waiting {wait}s...")
            time.sleep(wait)

        # Handle remaining APIStatusError exceptions,
        # usually NOT worth blind retrying
        except APIStatusError as e:
            print(f"[{model}] Client error (not retryable): {e}")
            return None

        # Handle APIConnectionError exceptions (network issues, timeouts)
        except APIConnectionError as e:
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Connection error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

        # Handle any other unexpected exceptions that may occur during the API call
        except Exception as e:
            wait = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Unexpected error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    # If we reach this point, it means all retries have been exhausted without a successful response, so we log that and return None.
    print(f"[{model}] Max retries exceeded. Giving up on this model.")
    return None
