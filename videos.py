"""
videos.py
=========

Writes ONE MP4 per model, streaming frames to disk as they are produced so the
run never holds a whole video in memory - which is what makes high-resolution
output affordable. Each trial gets a short title card and its last frame is held
for a moment so you can see the outcome.

    w = VideoWriter(path, fps=6, frame_size=(960, 960))
    w.title(["gpt-4o-mini", "trial 1"])
    for frame in frames: w.frame(frame)
    w.hold(frames[-1])
    w.close()

Requires imageio with the ffmpeg plugin (imageio-ffmpeg). If that's missing the
writer disables itself and the run continues without a video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("crafter_experiment.video")

TITLE_HOLD_FRAMES = 6   # how long a trial's title card stays up
END_HOLD_FRAMES = 8     # how long the final frame of a trial is held
BG = (18, 20, 26)       # title-card background (matches the viewers)
FG = (230, 232, 239)    # title-card text colour


# =============================================================================
#  Title cards
# =============================================================================
def _title_card(size: tuple[int, int], lines: list[str]) -> np.ndarray:
    """A solid card the same size as the frames, with centred, scaled text."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    font_size = max(14, min(w, h) // 14)      # scale text up on big frames
    try:
        font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()

    line_h = int(font_size * 1.4)
    y = (h - line_h * len(lines)) // 2
    for line in lines:
        try:
            tw = draw.textlength(line, font=font)
        except TypeError:
            tw = draw.textlength(line)
        draw.text(((w - tw) / 2, y), line, fill=FG, font=font)
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
#  Streaming writer
# =============================================================================
class VideoWriter:
    """Appends frames to an MP4 one at a time (constant memory)."""

    def __init__(self, out_path: Path, fps: int, frame_size: tuple[int, int]):
        self._out = Path(out_path)
        self._fps = int(fps)
        self._w, self._h = int(frame_size[0]), int(frame_size[1])
        self._writer = None
        self._count = 0
        self._failed = False

    def _ensure(self) -> None:
        if self._writer is not None or self._failed:
            return
        try:
            import imageio.v2 as imageio
            self._out.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                self._out, fps=self._fps, codec="libx264",
                quality=8, macro_block_size=1,
            )
        except Exception as exc:  # missing ffmpeg plugin, etc.
            self._failed = True
            LOG.error(
                "video disabled (%s). Install it with: pip install imageio-ffmpeg",
                exc,
            )

    def _append(self, frame: np.ndarray) -> None:
        self._ensure()
        if self._failed:
            return
        self._writer.append_data(_even(np.asarray(frame, dtype=np.uint8)))
        self._count += 1

    # -- public API -----------------------------------------------------------
    def title(self, lines: list[str], hold: int = TITLE_HOLD_FRAMES) -> None:
        card = _title_card((self._w, self._h), lines)
        for _ in range(hold):
            self._append(card)

    def frame(self, arr: np.ndarray) -> None:
        self._append(arr)

    def hold(self, arr: np.ndarray, n: int = END_HOLD_FRAMES) -> None:
        for _ in range(n):
            self._append(arr)

    def close(self) -> Path | None:
        if self._writer is not None:
            self._writer.close()
        return self._out if self._count > 0 else None
