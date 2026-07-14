# ============================================================
# WORLD CONFIGURATION
# Edit this file only.
# ============================================================

WORLD_SIZE = 10
VIEW_SIZE = 9
WINDOW_SIZE = 900
SEED = 1

# False creates a blank world using BACKGROUND.
# True keeps Crafter's normal procedural generation.
RANDOM_WORLD = False

# False disables Crafter's automatic mob spawning.
NATURAL_MOBS = False

BACKGROUND = "grass"

# Exact player position.
PLAYER = (5, 5)

# Action applied to the player on every turn.
# The keyboard does not control the player.
#
# Examples:
#   "noop"
#   "move_left"
#   "move_right"
#   "move_up"
#   "move_down"
#   "do"
#   "sleep"
PLAYER_ACTION = "noop"

# Maximum number of turns.
# Use None for no limit.
MAX_TURNS = 100

# Exact single tiles.
#
# Format:
#   (x, y): "tile"
TILES = {
    (0, 0): "tree",
    (8, 5): "tree",
    (9, 5): "tree",
}

# Rectangular areas.
#
# Format:
#   ("tile", x, y, width, height)
RECTANGLES = [
    ("water", 1, 3, 3, 4),
]

# Exact mob positions.
#
# Available:
#   "cow"
#   "zombie"
#   "skeleton"
#
# Format:
#   ("mob", x, y)
MOBS = [
    ("cow", 7, 7),
]
