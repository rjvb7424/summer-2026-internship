"""
videos.py
=========

Turns the per-turn frames captured during a run into ONE MP4 per model, so the
run leaves a single watchable video instead of a folder full of screenshots.

Frames are held in memory during the run (never written as individual PNGs) and
handed here trial by trial. Each trial is topped with a short title card and its
last frame is held for a moment so you can see the outcome.

    build_model_video("gpt-4o-mini", trials, Path("runs/x/videos/gpt.mp4"), fps=4)

``trials`` is a list of ``(label, frames)`` where ``frames`` is a list of
HxWx3 uint8 arrays. Requires imageio with the ffmpeg plugin (imageio-ffmpeg).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TITLE_HOLD_FRAMES = 6   # how long a trial's title card stays up
END_HOLD_FRAMES = 8     # how long the final frame of a trial is held
BG = (18, 20, 26)       # title-card background (matches the viewers)
FG = (230, 232, 239)    # title-card text colour


# =============================================================================
#  Title cards
# =============================================================================
def _title_card(size: tuple[int, int], lines: list[str]) -> np.ndarray:
    """A solid card the same size as the frames, with centred text."""
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    # Default bitmap font keeps this dependency-free and legible at small sizes.
    line_h = 14
    total = line_h * len(lines)
    y = (h - total) // 2
    for line in lines:
        tw = draw.textlength(line)
        draw.text(((w - tw) / 2, y), line, fill=FG)
        y += line_h
    return np.asarray(img)


def _even(frame: np.ndarray) -> np.ndarray:
    """Pad to even width/height - libx264/yuv420p needs even dimensions."""
    h, w = frame.shape[:2]
    ph, pw = h % 2, w % 2
    if ph or pw:
        frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return frame


# =============================================================================
#  Video assembly
# =============================================================================
def build_model_video(
    model_name: str,
    trials: list[tuple[str, list[np.ndarray]]],
    out_path: Path,
    fps: int = 4,
) -> Path | None:
    """Write one MP4 covering all of a model's trials. Returns the path, or
    None if there was nothing to render."""
    import imageio.v2 as imageio

    frame_lists = [f for _, f in trials if f]
    if not frame_lists:
        return None
    h, w = frame_lists[0][0].shape[:2]

    sequence: list[np.ndarray] = []
    for label, frames in trials:
        if not frames:
            continue
        card = _title_card((w, h), [model_name, label])
        sequence.extend([card] * TITLE_HOLD_FRAMES)
        sequence.extend(frames)
        sequence.extend([frames[-1]] * END_HOLD_FRAMES)

    sequence = [_even(np.asarray(f, dtype=np.uint8)) for f in sequence]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 keeps our exact dimensions (we already made them even).
    with imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        quality=8, macro_block_size=1,
    ) as writer:
        for frame in sequence:
            writer.append_data(frame)
    return out_path
