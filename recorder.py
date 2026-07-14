"""Video recorder: one .mp4 per trial."""

from pathlib import Path

import imageio

import config


class Recorder:
    """Writes RGB frames to an mp4 file."""

    def __init__(self, model_name, trial_index):
        safe_name = model_name.replace("/", "__")
        directory = Path(config.RECORDINGS_DIR) / safe_name
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"trial_{trial_index:03d}.mp4"
        self.writer = imageio.get_writer(self.path, fps=config.VIDEO_FPS, macro_block_size=1)

    def add_frame(self, frame):
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()