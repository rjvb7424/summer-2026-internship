from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pygame

import config
from crafter_env import FixedCrafterWorld
from hf_agent import HuggingFaceAgent


def draw_world(
    screen,
    game,
    trial_number,
    turn,
    action,
    success,
):
    frame = game.render()
    surface = pygame.surfarray.make_surface(
        frame.swapaxes(0, 1)
    )
    screen.blit(surface, (0, 0))

    panel = pygame.Surface((config.WINDOW_SIZE, 115))
    panel.set_alpha(220)
    panel.fill((0, 0, 0))
    screen.blit(panel, (0, 0))

    font = pygame.font.Font(None, 28)
    small = pygame.font.Font(None, 23)

    title = font.render(
        f"Trial {trial_number}/{config.NUM_TRIALS} | Turn {turn}",
        True,
        (255, 255, 255),
    )
    screen.blit(title, (16, 12))

    state = game.state()
    lines = [
        f"Goal: {config.GOAL_DESCRIPTION}",
        f"Action: {action}",
        (
            f"Position: {state['position']} | "
            f"Facing: {state['facing']} | "
            f"Wood: {state['inventory'].get('wood', 0)}"
        ),
    ]

    y = 43
    for line in lines:
        text = small.render(line, True, (235, 235, 235))
        screen.blit(text, (16, y))
        y += 22

    if success:
        text = font.render(
            "ACHIEVEMENT UNLOCKED",
            True,
            (255, 235, 120),
        )
        screen.blit(text, (config.WINDOW_SIZE - 275, 15))

    pygame.display.flip()


def process_window_events() -> bool:
    """Return False when the user closes the visualizer."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
        ):
            return False
    return True


def save_results(results):
    path = Path(config.RESULTS_FILE)
    path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def run_trial(
    agent,
    trial_index,
    screen,
):
    seed = config.BASE_SEED + trial_index
    game = FixedCrafterWorld(seed=seed)

    history = []
    total_reward = 0.0
    success = False
    stopped_by_user = False
    start_time = time.perf_counter()

    if screen is not None:
        draw_world(
            screen,
            game,
            trial_index + 1,
            turn=0,
            action="not started",
            success=False,
        )

    # Turn counting and experiment flow live here in main.py.
    for turn in range(1, config.MAX_TURNS + 1):
        if screen is not None and not process_window_events():
            stopped_by_user = True
            break

        observation = game.observation_text(
            turn=turn,
            history=history,
        )

        action, raw_response = agent.choose_action(observation)

        _, reward, env_done, info = game.execute(action)
        total_reward += float(reward)

        wood = int(game.player.inventory.get("wood", 0))
        success = game.has_achievement(config.GOAL_ACHIEVEMENT)

        step_record = {
            "turn": turn,
            "action": action,
            "raw_response": raw_response,
            "reward": float(reward),
            "wood": wood,
            "position": [
                int(value) for value in game.player.pos
            ],
            "facing": game.facing_name(),
            "achievement_unlocked": success,
        }
        history.append(step_record)

        print(
            f"[trial {trial_index + 1}] "
            f"turn={turn:02d} action={action:<10} "
            f"position={tuple(game.player.pos)} "
            f"facing={game.facing_name():<5} wood={wood}"
        )
        print(f"  model: {raw_response!r}")

        if screen is not None:
            draw_world(
                screen,
                game,
                trial_index + 1,
                turn,
                action,
                success,
            )
            pygame.time.wait(config.TURN_DELAY_MS)

        if success or env_done:
            break

    elapsed = time.perf_counter() - start_time

    return {
        "trial": trial_index + 1,
        "seed": seed,
        "model": config.MODEL_NAME,
        "goal": config.GOAL_ACHIEVEMENT,
        "success": success,
        "turns": len(history),
        "total_reward": total_reward,
        "elapsed_seconds": elapsed,
        "stopped_by_user": stopped_by_user,
        "steps": history,
    }


def run():
    screen = None

    if config.VISUALIZE:
        pygame.init()
        screen = pygame.display.set_mode(
            (config.WINDOW_SIZE, config.WINDOW_SIZE)
        )
        pygame.display.set_caption(
            "Crafter Hugging Face wood experiment"
        )

    agent = HuggingFaceAgent()
    results = []

    try:
        for trial_index in range(config.NUM_TRIALS):
            result = run_trial(
                agent=agent,
                trial_index=trial_index,
                screen=screen,
            )
            results.append(result)
            save_results(results)

            status = "SUCCESS" if result["success"] else "FAILED"
            print(
                f"\nTrial {result['trial']}: {status} "
                f"after {result['turns']} turns "
                f"({result['elapsed_seconds']:.1f}s)\n"
            )

            if result["stopped_by_user"]:
                break
    finally:
        save_results(results)
        if config.VISUALIZE:
            pygame.quit()

    successes = sum(result["success"] for result in results)
    print(
        f"Finished: {successes}/{len(results)} successful trials."
    )
    print(f"Results saved to {config.RESULTS_FILE}")


if __name__ == "__main__":
    run()
