import time

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer


_MODEL_CACHE = {}
MIN_TRANSFORMERS_VERSION = Version("4.49.0")


def validate_transformers_version():
    """Ensure the installed Transformers version supports Phi-4 natively."""
    installed_version = Version(transformers.__version__)
    if installed_version < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Phi-4 Mini requires transformers>=4.49.0. "
            f"Installed version: {installed_version}"
        )


def get_model(model_name):
    """Load and cache the tokenizer and causal language model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    validate_transformers_version()
    print(f"[{model_name}] Loading model into memory...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        max_memory={
            "mps": "10GiB",
            "cpu": "2GiB",
        },
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _MODEL_CACHE[model_name] = (model, tokenizer)
    return _MODEL_CACHE[model_name]


def _model_input_device(model):
    """Return the device that should receive the input token tensors."""
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _prompt_input_ids(tokenizer, messages):
    """Apply the model chat template and return one prompt token tensor."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]

    return encoded


def _score_candidate(model, tokenizer, prompt_ids, candidate_action):
    """Return the average log probability of one candidate continuation."""
    candidate_ids = tokenizer(
        candidate_action,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    device = _model_input_device(model)
    prompt_ids = prompt_ids.to(device)
    candidate_ids = candidate_ids.to(device)
    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    score = -float(outputs.loss.detach().cpu())
    return score, int(candidate_ids.shape[1])


def call_huggingface(
    messages,
    model,
    candidate_actions=None,
    max_retries=3,
):
    """Choose the most likely action from a state-aware candidate list.

    This deliberately scores complete action names instead of generating free
    text. It removes malformed outputs, generation-length warnings, and random
    action drift while still letting the language model make the selection.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if not candidate_actions:
        raise ValueError(
            "candidate_actions is required for constrained Crafter decisions."
        )

    candidate_actions = list(dict.fromkeys(candidate_actions))
    model_object, tokenizer = get_model(model)
    prompt_ids = _prompt_input_ids(tokenizer, messages)
    prompt_tokens = int(prompt_ids.shape[-1])

    if len(candidate_actions) == 1:
        selected_action = candidate_actions[0]
        output_tokens = len(
            tokenizer.encode(
                selected_action,
                add_special_tokens=False,
            )
        )
        print(
            f"[{model}] Action mask selected the only valid strategic "
            f"action: {selected_action}"
        )
        return {
            "text": selected_action,
            "elapsed_seconds": 0.0,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": None,
            "total_tokens": prompt_tokens + output_tokens,
            "finish_reason": "single_candidate",
            "model_version": model,
            "is_partial": False,
            "candidate_actions": candidate_actions,
            "candidate_scores": {selected_action: None},
            "decision_mode": "forced_single_candidate",
        }

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            print(
                f"[{model}] [Attempt {attempt}] Scoring candidates: "
                f"{', '.join(candidate_actions)}"
            )

            scores = {}
            output_token_counts = {}
            for action in candidate_actions:
                score, output_tokens = _score_candidate(
                    model=model_object,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    candidate_action=action,
                )
                scores[action] = score
                output_token_counts[action] = output_tokens

            selected_action = max(scores, key=scores.get)
            elapsed = time.time() - start
            selected_output_tokens = output_token_counts[selected_action]

            score_summary = ", ".join(
                f"{action}={score:.3f}"
                for action, score in sorted(
                    scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            print(
                f"[{model}] Selected {selected_action} after "
                f"{elapsed:.1f}s | {score_summary}"
            )

            return {
                "text": selected_action,
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "output_tokens": selected_output_tokens,
                "thinking_tokens": None,
                "total_tokens": prompt_tokens + selected_output_tokens,
                "finish_reason": "candidate_scoring",
                "model_version": model,
                "is_partial": False,
                "candidate_actions": candidate_actions,
                "candidate_scores": scores,
                "decision_mode": "model_scored_candidates",
            }

        except Exception as error:
            if attempt == max_retries:
                print(f"[{model}] Candidate scoring failed: {error}")
                raise

            wait_seconds = 2 ** attempt
            print(
                f"[{model}] [Attempt {attempt}] Error: {error}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    return None
