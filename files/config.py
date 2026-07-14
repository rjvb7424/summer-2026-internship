"""Central configuration. Every experiment flag lives here."""

# ============================================================
# Models
# ============================================================
# Toggle providers on/off without touching the model lists.
ENABLE_HUGGINGFACE = True
ENABLE_OPENAI = True          # needs OPENAI_API_KEY
ENABLE_GEMINI = True          # needs GEMINI_API_KEY or GOOGLE_API_KEY

HUGGINGFACE_MODELS = [
    "microsoft/Phi-4-mini-instruct",
    "microsoft/phi-2",
]
OPENAI_MODELS = [
    "gpt-4o-mini",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
]

# Run order: enabled providers expand into (provider, model) pairs.
MODELS = (
    [("huggingface", m) for m in HUGGINGFACE_MODELS if ENABLE_HUGGINGFACE]
    + [("openai", m) for m in OPENAI_MODELS if ENABLE_OPENAI]
    + [("gemini", m) for m in GEMINI_MODELS if ENABLE_GEMINI]
)

# ============================================================
# Experiment
# ============================================================
TRIALS_PER_MODEL = 3          # trials per model
MAX_STEPS_PER_TRIAL = 300     # hard step cap per trial
BASE_SEED = 42                # trial i uses seed BASE_SEED + i (same seeds across models)

# ============================================================
# Agent / generation (shared)
# ============================================================
HISTORY_LENGTH = 6            # past (action -> outcome) pairs kept in the prompt
LOCAL_VIEW_RADIUS = 4         # half-width of the semantic window described to the LLM

# ---- HuggingFace (local) ----
MAX_NEW_TOKENS = 24           # short: we only need one action name
DO_SAMPLE = False             # False = greedy decoding (deterministic)
TEMPERATURE = 0.7             # only used when DO_SAMPLE is True

# ---- API providers (OpenAI / Gemini) ----
API_MAX_TOKENS = 2048         # generous: reasoning models spend tokens thinking
API_TEMPERATURE = 0.0         # deterministic where the model allows it
API_TIMEOUT_SEC = 60          # per-call timeout
API_MAX_RETRIES = 3
API_RETRY_DELAY_SEC = 2       # backoff base: delay * attempt

# ============================================================
# Hardware (Apple Silicon friendly)
# ============================================================
# Cap MPS memory so the run never hard-crashes the machine.
# High = hard cap (share of unified memory); low = GC trigger, must be <= high.
# 0.8 keeps ~20% free for macOS + your editor while the experiment stays priority.
MPS_HIGH_WATERMARK_RATIO = "0.8"
MPS_LOW_WATERMARK_RATIO = "0.7"
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