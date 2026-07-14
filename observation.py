"""
observation.py
==============

Turns the live Crafter world into the two things the experiment consumes:

* a **text observation** (ASCII map + legend + inventory + achievements) that is
  fed to the language model, and
* a **top-down PNG image** of the whole world for the human viewer.

This module is AI-facing (its text output goes into the prompt), so it contains
no prints, inputs, or side effects beyond the optional image save.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
#  Symbol tables  (edit these to change how the model "sees" the world)
# =============================================================================
MATERIAL_SYMBOLS: dict[str | None, str] = {
    None: " ",
    "grass": ".",
    "tree": "T",
    "water": "~",
    "stone": "#",
    "path": "-",
    "sand": ":",
    "coal": "c",
    "iron": "i",
    "diamond": "*",
    "lava": "!",
    "table": "=",
    "furnace": "U",
}

OBJECT_SYMBOLS: dict[str, str] = {
    "Player": "@",
    "Cow": "C",
    "Zombie": "Z",
    "Skeleton": "K",
    "Plant": "p",
    "Arrow": ">",
}

MATERIAL_NAMES: dict[str, str] = {
    ".": "grass", "T": "tree", "~": "water", "#": "stone", "-": "path",
    ":": "sand", "c": "coal", "i": "iron", "*": "diamond", "!": "lava",
    "=": "crafting table", "U": "furnace",
}

OBJECT_NAMES: dict[str, str] = {
    "@": "you (the player)", "C": "cow", "Z": "zombie",
    "K": "skeleton", "p": "plant", ">": "arrow",
}

FACING_NAMES: dict[tuple[int, int], str] = {
    (-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down",
}


# =============================================================================
#  Text observation helpers
# =============================================================================
def render_text_map(world, player) -> str:
    """Return the world as rows of single characters (objects override tiles)."""
    width, height = world.area
    player_pos = tuple(player.pos)
    rows: list[str] = []
    for y in range(height):
        row_chars: list[str] = []
        for x in range(width):
            material, obj = world[x, y]
            if (x, y) == player_pos:
                row_chars.append(OBJECT_SYMBOLS["Player"])
            elif obj is not None:
                row_chars.append(OBJECT_SYMBOLS.get(type(obj).__name__, "?"))
            else:
                row_chars.append(MATERIAL_SYMBOLS.get(material, "?"))
        rows.append("".join(row_chars))
    return "\n".join(rows)


def build_legend(world, player) -> str:
    """Explain only the symbols actually present on the current map."""
    present: set[str] = set()
    width, height = world.area
    player_pos = tuple(player.pos)
    for y in range(height):
        for x in range(width):
            material, obj = world[x, y]
            if (x, y) == player_pos:
                present.add(OBJECT_SYMBOLS["Player"])
            elif obj is not None:
                present.add(OBJECT_SYMBOLS.get(type(obj).__name__, "?"))
            else:
                present.add(MATERIAL_SYMBOLS.get(material, "?"))

    lines: list[str] = []
    for symbol in sorted(present):
        name = OBJECT_NAMES.get(symbol) or MATERIAL_NAMES.get(symbol)
        if name:
            lines.append(f"  {symbol} = {name}")
    return "\n".join(lines)


def format_inventory(inventory: dict[str, int]) -> str:
    """List only items the player actually holds."""
    held = [f"  {name}: {count}" for name, count in inventory.items() if count > 0]
    return "\n".join(held) if held else "  (empty)"


def format_achievements(achievements: dict[str, int]) -> str:
    """Comma-separated list of unlocked achievement names, or 'none yet'."""
    unlocked = [name for name, count in achievements.items() if count > 0]
    return ", ".join(unlocked) if unlocked else "none yet"


def describe_position(player) -> str:
    x, y = player.pos
    return f"(x={int(x)}, y={int(y)})"


def describe_facing(player) -> str:
    return FACING_NAMES.get(tuple(player.facing), "unknown")


# =============================================================================
#  Image renderer  (top-down view of the whole small world)
# =============================================================================
class ImageRenderer:
    """
    Draws the entire world top-down using Crafter's own tile textures.

    Unlike ``env.render()`` (an egocentric 9x9 crop), this shows every tile,
    which is what you want for a tiny hand-built world.
    """

    def __init__(self, textures, tile_pixels: int = 24):
        self._textures = textures
        self._unit = int(tile_pixels)

    # -- public API -----------------------------------------------------------
    def render_array(self, world, player) -> np.ndarray:
        """Return an ``(H*unit, W*unit, 3)`` uint8 RGB image."""
        width, height = world.area
        unit = self._unit
        canvas = np.zeros((width * unit, height * unit, 3), np.uint8) + 127

        for x in range(width):
            for y in range(height):
                material = world[x, y][0]
                self._blit(canvas, x, y, self._tex(material))

        for obj in world.objects:
            x, y = int(obj.pos[0]), int(obj.pos[1])
            self._blit(canvas, x, y, self._tex(obj.texture), alpha=True)

        # Transpose so x is horizontal and y is vertical when displayed.
        return canvas.transpose((1, 0, 2))

    def save(self, world, player, path: str | Path) -> Path:
        """Render the world and write it to ``path`` as a PNG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.render_array(world, player)).save(path)
        return path

    # -- internals ------------------------------------------------------------
    def _tex(self, name) -> np.ndarray:
        return self._textures.get(name, (self._unit, self._unit))

    def _blit(self, canvas, x, y, texture, alpha: bool = False) -> None:
        unit = self._unit
        px, py = x * unit, y * unit
        if alpha and texture.shape[-1] == 4:
            a = texture[..., 3:].astype(np.float32) / 255
            rgb = texture[..., :3].astype(np.float32) / 255
            cur = canvas[px:px + unit, py:py + unit].astype(np.float32) / 255
            canvas[px:px + unit, py:py + unit] = (255 * (a * rgb + (1 - a) * cur)).astype(np.uint8)
        else:
            canvas[px:px + unit, py:py + unit] = texture[..., :3]
