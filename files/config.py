"""Central configuration. Every experiment flag lives here."""

# ============================================================
# Models
# ============================================================
# HuggingFace model ids, run in order. Add or remove freely.
MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/deepseek-llm-7b-chat",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]

# ============================================================
# Experiment
# ============================================================
TRIALS_PER_MODEL = 3          # trials per model
MAX_STEPS_PER_TRIAL = 300     # hard step cap per trial
BASE_SEED = 42                # trial i uses seed BASE_SEED + i (same seeds across models)

# ============================================================
# Agent / generation
# ============================================================
MAX_NEW_TOKENS = 24           # short: we only need one action name
DO_SAMPLE = False             # False = greedy decoding (deterministic)
TEMPERATURE = 0.7             # only used when DO_SAMPLE is True
HISTORY_LENGTH = 6            # past (action -> outcome) pairs kept in the prompt
LOCAL_VIEW_RADIUS = 4         # half-width of the semantic window described to the LLM

# ============================================================
# Hardware (Apple Silicon friendly)
# ============================================================
# Cap MPS memory so the run never hard-crashes the machine.
# High = hard cap (0.85 = 85% of unified memory); low = GC trigger, must be <= high.
MPS_HIGH_WATERMARK_RATIO = "0.85"
MPS_LOW_WATERMARK_RATIO = "0.75"
TORCH_DTYPE = "float16"       # fp16 on MPS; falls back to float32 on CPU

# ============================================================
# Viewer / recording
# ============================================================
SHOW_VIEWER = True            # live pygame window
VIEWER_SIZE = 512             # viewer render size (pixels, square)
RECORD_VIDEO = True           # save an .mp4 per trial
VIDEO_SIZE = 320              # video frame size (pixels, square)
VIDEO_FPS = 8

# ============================================================
# Paths
# ============================================================
RESULTS_DIR = "results"
RECORDINGS_DIR = "recordings"
GRAPHS_DIR = "results/graphs"