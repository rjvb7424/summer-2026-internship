"""Crafter environment wrapper: steps the game and renders text observations."""

import itertools

import crafter
import numpy as np

import config

# ============================================================
# Constants
# ============================================================
ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]

STATS = ("health", "food", "drink", "energy")


# ============================================================
# Session
# ============================================================
class CrafterSession:
    """One Crafter episode with text observation support."""

    def __init__(self, seed):
        self.env = crafter.Env(seed=seed)
        self.id_to_name = self._build_id_map(self.env)
        self.last_info = None

    # -------------------- lifecycle --------------------

    def reset(self):
        """Start the episode and return the initial text observation."""
        self.env.reset()
        # A noop step populates the info dict (inventory, semantic map, ...).
        _, _, _, info = self.env.step(0)
        self.last_info = info
        return self.text_observation()

    def step(self, action_name):
        """Apply an action by name. Returns (text_obs, reward, done, info)."""
        action_index = ACTION_NAMES.index(action_name)
        _, reward, done, info = self.env.step(action_index)
        self.last_info = info
        return self.text_observation(), reward, done, info

    def render_frame(self, size):
        """RGB frame of the current state at the given square pixel size."""
        return self.env.render(size=(size, size))

    # -------------------- observation --------------------

    def text_observation(self):
        """Compact natural-language description of the current state."""
        info = self.last_info
        surroundings = self._describe_surroundings(info)
        stats = ", ".join(f"{name}: {info['inventory'][name]}/9" for name in STATS)
        items = {
            name: count for name, count in info["inventory"].items()
            if name not in STATS and count > 0
        }
        inventory = (
            ", ".join(f"{name} x{count}" for name, count in items.items())
            or "empty"
        )
        return (
            f"Surroundings:\n{surroundings}\n"
            f"Stats: {stats}\n"
            f"Inventory: {inventory}"
        )

    def achievements(self):
        """Achievement name -> count for the current episode."""
        return dict(self.last_info["achievements"])

    # -------------------- internals --------------------

    @staticmethod
    def _build_id_map(env):
        """Map semantic ids to readable names (materials and objects)."""
        pairs = itertools.chain(
            env._world._mat_ids.items(), env._sem_view._obj_ids.items()
        )
        id_to_name = {}
        for key, index in pairs:
            if key is None:
                name = "void"
            elif isinstance(key, str):
                name = key
            else:  # object class, e.g. crafter.objects.Cow
                name = key.__name__.lower()
            id_to_name[index] = name
        return id_to_name

    def _describe_surroundings(self, info):
        """Nearest instance of each visible thing, with direction and distance."""
        semantic = info["semantic"]
        px, py = info["player_pos"]
        radius = config.LOCAL_VIEW_RADIUS
        nearest = {}
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if dx == dy == 0:
                    continue
                if not (0 <= x < semantic.shape[0] and 0 <= y < semantic.shape[1]):
                    continue
                name = self.id_to_name.get(semantic[x, y], "unknown")
                if name in ("void", "player", "unknown"):
                    continue
                distance = abs(dx) + abs(dy)
                if name not in nearest or distance < nearest[name][0]:
                    nearest[name] = (distance, dx, dy)
        if not nearest:
            return "- nothing notable nearby"
        lines = []
        for name, (_, dx, dy) in sorted(nearest.items(), key=lambda kv: kv[1][0]):
            lines.append(f"- {name}: {self._direction(dx, dy)}")
        return "\n".join(lines)

    @staticmethod
    def _direction(dx, dy):
        """Human-readable relative direction, e.g. '2 north, 1 east'."""
        parts = []
        if dy:
            parts.append(f"{abs(dy)} {'north' if dy < 0 else 'south'}")
        if dx:
            parts.append(f"{abs(dx)} {'west' if dx < 0 else 'east'}")
        return ", ".join(parts)
