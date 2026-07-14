# Experiment constants.
NUM_TRIALS = 5
MAX_STEPS = 500
BASE_SEED = 20260708

# File paths.
RESULTS_FILE = "results.json"
RECORD_DIRECTORY = "recordings"

# Toggle experiment features.
SHOW_SIMULATION = True
RECORD_VIDEO = True
SAVE_PROMPTS = False
RUN_ANALYSIS = True

# Toggle which AI providers to run in the experiment.
RUN_GEMINI = False
RUN_GPT = False
RUN_HUGGINGFACE = True

# AI models to use for the experiment.
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]
GPT_MODELS = [
    "o3-2025-04-16",
    "gpt-5-nano-2025-08-07",
    "gpt-5.5-2026-04-23",
    "gpt-5.4-mini-2026-03-17",
]
HUGGINGFACE_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/deepseek-llm-7b-chat",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]

# HuggingFace generation parameters.
MAX_NEW_TOKENS = 1024
MAX_MEMORY = {"mps": "10GiB", "cpu": "2GiB"}
