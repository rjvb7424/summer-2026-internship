import queue
import threading
import time
from collections import deque

import crafter

from crafter_env import player_position
from crafter_prompt import (
    ACTION_HISTORY_LIMIT,
    build_messages,
    describe_action_result,
    extract_action,
)
from viewer import CrafterViewer

# Execution configuration.
SOLVER_POLL_SECONDS = 0.05
RESPONSE_PREVIEW_LENGTH = 240


class CrafterTest:
    """Run one Crafter episode controlled by a prompt-based AI solver."""

    def __init__(
        self,
        max_steps=200,
        seed=0,
        show_simulation=True,
        record_directory=None,
        save_prompts=False,
    ):
        self.max_steps = max_steps
        self.seed = seed
        self.show_simulation = show_simulation
        self.record_directory = record_directory
        self.save_prompts = save_prompts

        self.env = crafter.Env(length=max_steps, seed=seed)
        if record_directory:
            self.env = crafter.Recorder(
                self.env,
                record_directory,
                save_stats=False,
                save_video=True,
                save_episode=False,
                video_size=(512, 512),
            )

    # -----------------------------------------------------------------------
    # Episode loop
    # -----------------------------------------------------------------------

    def run(self, solver, model="unknown", trial=1):
        """Run one episode and return a JSON-serializable result dictionary."""
        viewer = CrafterViewer() if self.show_simulation else None
        action_history = deque(maxlen=ACTION_HISTORY_LIMIT)
        trajectory = []

        total_reward = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        invalid_actions = 0
        solver_failed = False
        last_action = None
        last_action_result = "No previous action."
        last_reward = 0.0
        last_response = None
        done = False
        step = 0
        episode_start = time.time()

        self.env.reset()

        try:
            while not done and step < self.max_steps:
                messages = build_messages(
                    env=self.env,
                    step=step,
                    max_steps=self.max_steps,
                    action_history=action_history,
                    last_action=last_action,
                    last_action_result=last_action_result,
                    last_reward=last_reward,
                )

                solver_result = self._call_solver(
                    solver=solver,
                    messages=messages,
                    viewer=viewer,
                    model=model,
                    trial=trial,
                    step=step,
                    last_action=last_action,
                    last_reward=last_reward,
                    total_reward=total_reward,
                    last_response=last_response,
                )

                if not solver_result:
                    solver_failed = True
                    break

                raw_response = solver_result.get("text", "")
                action_name = extract_action(raw_response)
                if action_name is None:
                    action_name = "noop"
                    invalid_actions += 1

                position_before = player_position(self.env)
                inventory_before = self.env._player.inventory.copy()
                achievements_before = self.env._player.achievements.copy()
                facing_before = tuple(self.env._player.facing)

                action_index = self.env.action_names.index(action_name)
                _, reward, done, info = self.env.step(action_index)

                position_after = player_position(self.env)
                inventory_after = self.env._player.inventory.copy()
                achievements_after = self.env._player.achievements.copy()
                facing_after = tuple(self.env._player.facing)

                last_action_result = describe_action_result(
                    action=action_name,
                    position_before=position_before,
                    position_after=position_after,
                    facing_before=facing_before,
                    facing_after=facing_after,
                    inventory_before=inventory_before,
                    inventory_after=inventory_after,
                    achievements_before=achievements_before,
                    achievements_after=achievements_after,
                )

                step += 1
                last_action = action_name
                last_reward = float(reward)
                last_response = raw_response
                total_reward += float(reward)

                action_history.append({
                    "action": action_name,
                    "result": last_action_result,
                })

                total_prompt_tokens += solver_result.get("prompt_tokens") or 0
                total_output_tokens += solver_result.get("output_tokens") or 0
                total_tokens += solver_result.get("total_tokens") or 0

                decision = {
                    "step": step,
                    "action": action_name,
                    "action_result": last_action_result,
                    "reward": float(reward),
                    "position_before": list(position_before),
                    "position_after": list(position_after),
                    "raw_response": raw_response,
                    "elapsed_seconds": solver_result.get("elapsed_seconds"),
                    "prompt_tokens": solver_result.get("prompt_tokens"),
                    "output_tokens": solver_result.get("output_tokens"),
                    "thinking_tokens": solver_result.get("thinking_tokens"),
                    "total_tokens": solver_result.get("total_tokens"),
                    "is_partial": solver_result.get("is_partial", False),
                }
                if self.save_prompts:
                    decision["messages"] = messages
                trajectory.append(decision)

                if viewer:
                    viewer.render(
                        env=self.env,
                        model=model,
                        trial=trial,
                        step=step,
                        max_steps=self.max_steps,
                        status="Action completed",
                        last_action=last_action,
                        last_reward=last_reward,
                        total_reward=total_reward,
                        achievements=info["achievements"],
                        response_preview=self._response_preview(last_response),
                    )
        finally:
            if viewer:
                viewer.close()

        achievements = self.env._player.achievements.copy()
        inventory = self.env._player.inventory.copy()
        unlocked = [name for name, count in achievements.items() if count > 0]

        return {
            "model_version": model,
            "trial": trial,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "episode_steps": step,
            "total_reward": total_reward,
            "num_achievements": len(unlocked),
            "unlocked_achievements": unlocked,
            "achievements": achievements,
            "final_inventory": inventory,
            "invalid_actions": invalid_actions,
            "solver_failed": solver_failed,
            "elapsed_seconds": time.time() - episode_start,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "trajectory": trajectory,
        }

    # -----------------------------------------------------------------------
    # Solver execution
    # -----------------------------------------------------------------------

    def _call_solver(
        self,
        solver,
        messages,
        viewer,
        model,
        trial,
        step,
        last_action,
        last_reward,
        total_reward,
        last_response,
    ):
        """Call the solver without freezing the Pygame event loop."""
        if viewer is None:
            return solver(messages)

        result_queue = queue.Queue(maxsize=1)

        def run_solver():
            try:
                result_queue.put(("result", solver(messages)))
            except Exception as error:
                result_queue.put(("error", error))

        thread = threading.Thread(target=run_solver, daemon=True)
        thread.start()
        started = time.time()

        while thread.is_alive():
            elapsed = time.time() - started
            viewer.render(
                env=self.env,
                model=model,
                trial=trial,
                step=step,
                max_steps=self.max_steps,
                status=f"Model thinking... {elapsed:.1f}s",
                last_action=last_action,
                last_reward=last_reward,
                total_reward=total_reward,
                achievements=self.env._player.achievements,
                response_preview=self._response_preview(last_response),
            )
            time.sleep(SOLVER_POLL_SECONDS)

        kind, payload = result_queue.get()
        if kind == "error":
            raise payload
        return payload

    def _response_preview(self, response):
        """Return a compact model-response preview."""
        if not response:
            return None
        compact = " ".join(response.split())
        return compact[:RESPONSE_PREVIEW_LENGTH]