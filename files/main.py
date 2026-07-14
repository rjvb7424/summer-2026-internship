"""Entry point: runs every model for TRIALS_PER_MODEL trials, then analysis.

Usage: python main.py
Safe to interrupt; rerunning resumes from the last completed trial.
"""

import os
import subprocess
import sys
import time

import config

# Must be set before torch initialises the MPS allocator (hf_agent imports torch).
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = config.MPS_HIGH_WATERMARK_RATIO
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = config.MPS_LOW_WATERMARK_RATIO

from crafter_env import CrafterSession
from recorder import Recorder
from results_store import ResultsStore


# ============================================================
# Agent factory (lazy imports: only load the SDKs you use)
# ============================================================
def create_agent(provider, model_name):
    if provider == "huggingface":
        from hf_agent import HuggingFaceAgent
        return HuggingFaceAgent(model_name)
    if provider == "openai":
        from openai_agent import OpenAIAgent
        return OpenAIAgent(model_name)
    if provider == "gemini":
        from gemini_agent import GeminiAgent
        return GeminiAgent(model_name)
    raise ValueError(f"unknown provider: {provider}")


# ============================================================
# Single trial
# ============================================================
def run_trial(agent, trial_index, viewer):
    """Run one episode and return its result record."""
    seed = config.BASE_SEED + trial_index
    session = CrafterSession(seed=seed)
    observation = session.reset()
    agent.reset_history()
    recorder = Recorder(agent.model_name, trial_index) if config.RECORD_VIDEO else None

    total_reward = 0.0
    invalid_actions = 0
    actions = []
    start_time = time.time()
    step = 0
    try:
        for step in range(1, config.MAX_STEPS_PER_TRIAL + 1):
            action, _, valid = agent.choose_action(observation)
            invalid_actions += not valid
            observation, reward, done, _ = session.step(action)
            agent.record_outcome(action, reward)
            total_reward += reward
            actions.append(action)
            if recorder:
                recorder.add_frame(session.render_frame(config.VIDEO_SIZE))
            if viewer:
                viewer.update(
                    session.render_frame(config.VIEWER_SIZE),
                    [
                        f"model: {agent.model_name} ({agent.provider})",
                        f"trial: {trial_index + 1}/{config.TRIALS_PER_MODEL}   step: {step}/{config.MAX_STEPS_PER_TRIAL}",
                        f"action: {action}   reward: {total_reward:+.1f}",
                        f"health: {session.last_info['inventory']['health']}/9",
                    ],
                )
            if done:
                break
    finally:
        if recorder:
            recorder.close()

    achievements = session.achievements()
    return {
        "model": agent.model_name,
        "provider": agent.provider,
        "trial": trial_index,
        "seed": seed,
        "steps": step,
        "died": step < config.MAX_STEPS_PER_TRIAL,
        "total_reward": round(total_reward, 2),
        "achievements": achievements,
        "achievements_unlocked": sum(count > 0 for count in achievements.values()),
        "invalid_actions": invalid_actions,
        "actions": actions,
        "duration_sec": round(time.time() - start_time, 1),
    }


# ============================================================
# Experiment loop
# ============================================================
def main():
    store = ResultsStore()
    viewer = None
    if config.SHOW_VIEWER:
        from viewer import Viewer
        viewer = Viewer()

    try:
        for provider, model_name in config.MODELS:
            completed = store.completed_trials(model_name)
            if completed >= config.TRIALS_PER_MODEL:
                continue
            try:
                agent = create_agent(provider, model_name)
            except Exception as error:
                print(f"[skip] {model_name}: {error}")
                continue
            try:
                for trial_index in range(completed, config.TRIALS_PER_MODEL):
                    record = run_trial(agent, trial_index, viewer)
                    store.append_trial(model_name, record)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print(f"[abort model] {model_name}: {error}")
                # completed trials are saved; the run moves to the next model
            finally:
                agent.unload()
    except KeyboardInterrupt:
        pass  # partial results are already on disk; rerun to resume
    finally:
        if viewer:
            viewer.close()

    subprocess.run([sys.executable, "analyze_results.py"], check=False)


if __name__ == "__main__":
    main()