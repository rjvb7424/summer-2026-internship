"""Live pygame viewer: shows the game frame plus a status bar."""

import numpy as np

import config

STATUS_BAR_HEIGHT = 96
BACKGROUND = (18, 18, 18)
TEXT_COLOR = (230, 230, 230)


class Viewer:
    """A window displaying the current frame and experiment status."""

    def __init__(self):
        import pygame  # imported lazily so headless runs never touch pygame
        self.pygame = pygame
        pygame.init()
        size = (config.VIEWER_SIZE, config.VIEWER_SIZE + STATUS_BAR_HEIGHT)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Crafter LLM Benchmark")
        self.font = pygame.font.SysFont("menlo, monospace", 16)

    def update(self, frame, status_lines):
        """Draw one frame. Raises KeyboardInterrupt if the window is closed."""
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("viewer closed")
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (config.VIEWER_SIZE, config.VIEWER_SIZE))
        self.screen.fill(BACKGROUND)
        self.screen.blit(surface, (0, 0))
        for line_index, line in enumerate(status_lines[:4]):
            text = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (8, config.VIEWER_SIZE + 6 + line_index * 22))
        pygame.display.flip()

    def close(self):
        self.pygame.quit()
