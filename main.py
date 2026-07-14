import sys
import pygame
# Internal imports
import config
from crafter_world import CrafterWorld

def advance_turn(game):
    if config.PLAYER_ACTION not in game.env.action_names:
        raise ValueError(
            f"Unknown PLAYER_ACTION {config.PLAYER_ACTION!r}. "
            f"Available actions: {game.env.action_names}"
        )

    action_number = game.env.action_names.index(
        config.PLAYER_ACTION
    )

    _, reward, crafter_done, info = game.env.step(action_number)

    return float(reward), bool(crafter_done), info

def run():
    pygame.init()

    screen = pygame.display.set_mode((config.WINDOW_SIZE, config.WINDOW_SIZE))
    pygame.display.set_caption("Crafter Experiment")

    game = CrafterWorld()

    turn = 0
    last_reward = 0.0
    finished = False
    running = True

    print("SPACE: advance exactly one turn")
    print("R: reset the world")
    print("ESC: quit")
    print("The player is not keyboard-controlled.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    game.reset()
                    turn = 0
                    last_reward = 0.0
                    finished = False
                    print("World reset.")

                elif event.key == pygame.K_SPACE and not finished:
                    last_reward, crafter_done, info = advance_turn(
                        game
                    )
                    turn += 1

                    reached_limit = (
                        config.MAX_TURNS is not None
                        and turn >= config.MAX_TURNS
                    )

                    finished = crafter_done or reached_limit

                    print(
                        {
                            "turn": turn,
                            "reward": last_reward,
                            "finished": finished,
                            **game.get_state(),
                        }
                    )

        frame = game.render()
        surface = pygame.surfarray.make_surface(
            frame.swapaxes(0, 1)
        )

        screen.blit(surface, (0, 0))
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
