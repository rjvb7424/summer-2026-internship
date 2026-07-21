"""
main.py
=======

One command to run an experiment end to end:

    python main.py                 # run + LIVE view + plots + viewer
    python main.py my_config.yaml
    python main.py --no-live       # skip the real-time browser view
    python main.py --skip-analyze  # just run the trials
    python main.py --skip-run      # only (re)build plots + viewer from results

The live browser view is ON by default and opens a tab automatically. The
experiment is defined entirely by the YAML config; this file only wires the
pieces together, loads API keys from .env, and configures logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from config import load_config
from experiment import ExperimentRunner
from live_viewer import DEFAULT_PORT


# Third-party libraries that spam INFO logs, including HTTP request logs,
# urllib3 connection messages, Hugging Face messages, and similar output.
# These are pinned to WARNING so the console mainly shows experiment progress.
NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "urllib3.connectionpool",
    "huggingface_hub",
    "filelock",
    "openai",
    "google_genai",
    "google.genai",
]


def configure_logging() -> None:
    """Configure console logging for the experiment."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Silence the Hugging Face tokenizers fork warning.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_env() -> None:
    """Load API keys from a local .env file if python-dotenv is installed."""

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logging.getLogger("crafter_experiment").warning(
            "python-dotenv not installed - .env not loaded "
            "(pip install python-dotenv)."
        )


def get_experiment_name(results: dict, results_path: Path) -> str:
    """
    Return a readable experiment name for graph titles.

    For example:

        tree_opening_9x9

    becomes:

        tree opening 9x9
    """

    experiment = results.get("experiment", {})

    if isinstance(experiment, dict):
        name = experiment.get("name")
    else:
        name = experiment

    # Try alternative locations in case results.json uses a different format.
    name = (
        name
        or results.get("experiment_name")
        or results.get("name")
        or results_path.parent.name
        or "Experiment"
    )

    return str(name).replace("_", " ").strip()


def main() -> None:
    """Run the experiment, build plots, and create the replay viewer."""

    ap = argparse.ArgumentParser(
        description="Run a Crafter LLM experiment."
    )

    ap.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="path to the experiment config (default: config.yaml)",
    )

    ap.add_argument(
        "--no-live",
        action="store_true",
        help="disable the real-time browser view (on by default)",
    )

    ap.add_argument(
        "--no-browser",
        action="store_true",
        help="run the live view but do not auto-open a browser tab",
    )

    ap.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port for the live view (default: {DEFAULT_PORT})",
    )

    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="do not run trials",
    )

    ap.add_argument(
        "--skip-analyze",
        action="store_true",
        help="do not build plots",
    )

    ap.add_argument(
        "--skip-viewer",
        action="store_true",
        help="do not build viewer.html",
    )

    args = ap.parse_args()

    configure_logging()
    load_env()

    cfg = load_config(args.config)

    live = not args.no_live
    runner = None

    # -------------------------------------------------------------------------
    # Run experiment
    # -------------------------------------------------------------------------

    if not args.skip_run:
        runner = ExperimentRunner(
            cfg,
            live=live,
            live_port=args.port,
            open_browser=not args.no_browser,
        )

        runner.run()

    # -------------------------------------------------------------------------
    # Analyze results and build plots
    # -------------------------------------------------------------------------

    if not args.skip_analyze:
        import analyze_results

        results = json.loads(
            cfg.results_path.read_text(encoding="utf-8")
        )

        rows = analyze_results.summarise(results)

        experiment_name = get_experiment_name(
            results,
            cfg.results_path,
        )

        cfg.plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        analyze_results.plot_success_rate(
            rows,
            cfg.plots_dir / "success_rate.png",
            experiment_name,
        )

        analyze_results.plot_turns_to_success(
            rows,
            cfg.plots_dir / "turns_to_success.png",
            experiment_name,
        )

        analyze_results.plot_think_time(
            rows,
            cfg.plots_dir / "think_time.png",
            experiment_name,
        )

        analyze_results.plot_success_matrix(
            rows,
            cfg.plots_dir / "success_matrix.png",
            experiment_name,
        )

        analyze_results.print_summary(
            results,
            rows,
        )

        logging.getLogger("crafter_experiment").info(
            "Plots written to: %s",
            cfg.plots_dir,
        )

    # -------------------------------------------------------------------------
    # Build replay viewer
    # -------------------------------------------------------------------------

    if not args.skip_viewer:
        from viewer import build_viewer

        out = build_viewer(
            cfg.results_path,
            cfg.run_dir / "viewer.html",
        )

        logging.getLogger("crafter_experiment").info(
            "Replay viewer: %s",
            out,
        )

    # Keep the live server running so the final state remains visible until
    # the user closes the program.
    if (
        live
        and runner is not None
        and runner.live is not None
    ):
        runner.live.serve_until_interrupt()


if __name__ == "__main__":
    main()