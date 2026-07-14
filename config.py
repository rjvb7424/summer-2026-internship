# ============================================================
# EXPERIMENT CONFIGURATION
# Change this file to create new fixed-world experiments.
# ============================================================

# ----------------------------
# Hugging Face model
# ----------------------------

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# "auto" chooses MPS on Apple Silicon, CUDA when available, otherwise CPU.
DEVICE = "auto"

MAX_NEW_TOKENS = 24

# Greedy generation is easier to compare across repeated trials.
DO_SAMPLE = False
TEMPERATURE = 0.0

# ----------------------------
# Experiment
# ----------------------------

GOAL_ACHIEVEMENT = "collect_wood"
GOAL_DESCRIPTION = "Collect wood from one tree."

NUM_TRIALS = 5
MAX_TURNS = 20

# Change the seed between trials while keeping the fixed map unchanged.
# This mainly affects stochastic internal updates.
BASE_SEED = 100

# Save all model decisions and trial results here.
RESULTS_FILE = "wood_results.json"

# Show the world while the AI acts.
VISUALIZE = True
WINDOW_SIZE = 720

# Pause after each AI action so the result is visible.
TURN_DELAY_MS = 350

# ----------------------------
# Fixed Crafter world
# ----------------------------

WORLD_SIZE = 9
VIEW_SIZE = 9
BACKGROUND = "grass"

PLAYER_POSITION = (4, 4)

# Direction the player initially faces:
# "left", "right", "up", or "down"
PLAYER_FACING = "up"

# Exact material positions.
# The first tree is deliberately close to the player.
TILES = {
    (4, 2): "tree",
    (2, 3): "tree",
    (7, 6): "tree",
    (1, 7): "stone",
}

# Optional rectangles:
# ("material", x, y, width, height)
RECTANGLES = []

# This experiment contains no mobs.
MOBS = []

# Disable procedural generation and automatic spawning.
RANDOM_WORLD = False
NATURAL_MOBS = False

# ----------------------------
# Agent observation
# ----------------------------

# Include the complete map because this is a small fixed-world task.
SHOW_FULL_MAP = True

# Number of previous actions shown back to the model.
HISTORY_LENGTH = 8

VALID_ACTIONS = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "noop",
]
