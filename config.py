import os

# Cap MPS memory at 85% of unified memory so oversized models error instead of freezing.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.75")

# Experiment constants.
NUM_TRIALS = 6
MAX_STEPS = 500
BASE_SEED = 20260708

# File paths.
RESULTS_DIR = "results"
GRAPHS_DIR = "results/graphs"
RECORDINGS_DIR = "recordings"
VIDEO_FPS = 12

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
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]

# HuggingFace generation parameters.
MAX_NEW_TOKENS = 1024
MAX_MEMORY = {"mps": "10GiB", "cpu": "2GiB"}

# Constants for the window size and layout.
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
GAME_SIZE = 720
PANEL_WIDTH = WINDOW_WIDTH - GAME_SIZE
VIEWER_FPS = 12