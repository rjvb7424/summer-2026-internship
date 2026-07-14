"""Crafter episode runner: one trial per run(). Prompting lives in crafter_prompt."""

import crafter

import crafter_prompt
from recorder import Recorder
from crafter_viewer import CrafterViewer

HISTORY_LENGTH = 5


class CrafterTest:
    """Runs one Crafter trial where a language model chooses every action."""

    def __init__(self, max_steps, seed, show_simulation=False,
                 record_video=False, save_prompts=False):
        self.max_steps = max_steps
        self.seed = seed
        self.show_simulation = show_simulation
        self.record_video = record_video
        self.save_prompts = save_prompts

        self.env = crafter.Env(seed=seed)
        self.action_names = list(self.env.action_names)
        self.id_to_name = crafter_prompt.build_id_map(self.env)

    # ============================================================
    # Episode loop
    # ============================================================
    def run(self, solver, model, trial):
        """Run one full episode and return the trial record."""
        self.env.reset()
        _, _, _, info = self.env.step(self.action_names.index("noop"))

        viewer = CrafterViewer() if self.show_simulation else None
        recorder = Recorder(model, trial) if self.record_video else None

        total_reward = 0.0
        invalid_actions = 0
        steps = 0
        stopped_by_user = False
        history = []  # (action, result) pairs
        transcript = []
        last_action = None
        response_preview = None

        try:
            for step in range(1, self.max_steps + 1):
                prompt = crafter_prompt.build_user_prompt(
                    self.env, info, history[-HISTORY_LENGTH:], self.id_to_name,
                )

                if viewer and not viewer.render(
                    self.env, model, trial, step, self.max_steps, "Waiting for model...",
                    last_action, 0.0, total_reward, info["achievements"],
                    response_preview, prompt,
                ):
                    stopped_by_user = True
                    break

                response = solver(prompt, system_prompt=crafter_prompt.SYSTEM_PROMPT)
                text = response["text"] if response else ""
                action_index = crafter_prompt.parse_action(text, self.action_names)
                if action_index is None:
                    invalid_actions += 1
                    action_index = self.action_names.index("noop")
                last_action = self.action_names[action_index]
                response_preview = text.strip()

                if self.save_prompts:
                    transcript.append({"step": step, "prompt": prompt, "response": text})

                before = crafter_prompt.snapshot(self.env, info)
                _, reward, done, info = self.env.step(action_index)
                total_reward += reward
                steps = step
                history.append((
                    last_action,
                    crafter_prompt.describe_result(
                        last_action, before, crafter_prompt.snapshot(self.env, info)
                    ),
                ))

                if recorder:
                    recorder.add_frame(self.env.render((512, 512)))
                if viewer and not viewer.render(
                    self.env, model, trial, step, self.max_steps, "Acting",
                    last_action, reward, total_reward, info["achievements"],
                    response_preview, prompt,
                ):
                    stopped_by_user = True
                    break

                if done:
                    break
        finally:
            if recorder:
                recorder.close()
            if viewer:
                viewer.close()

        record = {
            "model_version": model,
            "trial": trial,
            "seed": self.seed,
            "steps": max(steps, 1),
            "total_reward": total_reward,
            "invalid_actions": invalid_actions,
            "achievements": info["achievements"],
            "achievements_unlocked": sum(
                1 for count in info["achievements"].values() if count > 0
            ),
            "stopped_by_user": stopped_by_user,
        }
        if self.save_prompts:
            record["transcript"] = transcript
        return record