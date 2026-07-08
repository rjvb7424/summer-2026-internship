import textwrap
import pygame

# Constants for the window size and layout.
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
GAME_SIZE = 720
PANEL_WIDTH = WINDOW_WIDTH - GAME_SIZE
VIEWER_FPS = 12


class CrafterViewer:
    """Display the current Crafter state while the AI chooses actions."""

    def __init__(self, title="Crafter AI Experiment"):
        pygame.init()
        pygame.display.set_caption(title)

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)
        self.running = True

    def process_events(self):
        """Process window events and return False when the user closes the viewer."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
        return self.running

    def render(
        self,
        env,
        model,
        trial,
        step,
        max_steps,
        status,
        last_action=None,
        last_reward=0.0,
        total_reward=0.0,
        achievements=None,
        response_preview=None,
    ):
        """Render the game and experiment information in one responsive window."""
        if not self.process_events():
            return False

        game_image = env.render((GAME_SIZE, GAME_SIZE))
        game_surface = pygame.surfarray.make_surface(
            game_image.transpose((1, 0, 2))
        )

        self.screen.fill((22, 24, 28))
        self.screen.blit(game_surface, (0, 0))
        pygame.draw.rect(
            self.screen,
            (34, 37, 43),
            pygame.Rect(GAME_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT),
        )

        y = 24
        y = self._draw_text("AI CONTROL PANEL", GAME_SIZE + 20, y, self.font)
        y += 12
        y = self._draw_wrapped(f"Model: {model}", GAME_SIZE + 20, y)
        y = self._draw_text(
            f"Trial: {trial}    Step: {step}/{max_steps}",
            GAME_SIZE + 20,
            y,
            self.small_font,
        )
        y += 10
        y = self._draw_wrapped(f"Status: {status}", GAME_SIZE + 20, y)
        y += 8
        y = self._draw_wrapped(
            f"Last action: {last_action or '-'}", GAME_SIZE + 20, y
        )
        y = self._draw_text(
            f"Last reward: {last_reward:.2f}",
            GAME_SIZE + 20,
            y,
            self.small_font,
        )
        y = self._draw_text(
            f"Total reward: {total_reward:.2f}",
            GAME_SIZE + 20,
            y,
            self.small_font,
        )

        y += 18
        y = self._draw_text("Achievements", GAME_SIZE + 20, y, self.font)
        unlocked = [
            name for name, count in (achievements or {}).items() if count > 0
        ]
        achievement_text = ", ".join(unlocked) if unlocked else "None yet"
        y = self._draw_wrapped(achievement_text, GAME_SIZE + 20, y)

        if response_preview:
            y += 18
            y = self._draw_text("Model response", GAME_SIZE + 20, y, self.font)
            self._draw_wrapped(response_preview, GAME_SIZE + 20, y, max_lines=8)

        pygame.display.flip()
        self.clock.tick(VIEWER_FPS)
        return True

    def close(self):
        """Close the Pygame window."""
        pygame.quit()

    def _draw_text(self, text, x, y, font):
        surface = font.render(text, True, (235, 238, 242))
        self.screen.blit(surface, (x, y))
        return y + surface.get_height() + 5

    def _draw_wrapped(self, text, x, y, max_lines=None):
        width = max(20, (PANEL_WIDTH - 40) // 10)
        lines = textwrap.wrap(str(text), width=width) or [""]
        if max_lines is not None:
            lines = lines[:max_lines]

        for line in lines:
            y = self._draw_text(line, x, y, self.small_font)
        return y
