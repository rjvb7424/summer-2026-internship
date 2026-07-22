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

from config import load_config
from experiment import ExperimentRunner
from live_viewer import DEFAULT_PORT

# Third-party libraries that spam INFO logs (the "HTTP Request: GET ..." lines,
# urllib3 connection chatter, etc.). Pinned to WARNING so the console only shows
# our own trial progress.
NOISY_LOGGERS = [
    "httpx", "httpcore", "urllib3", "urllib3.connectionpool",
    "huggingface_hub", "filelock", "openai", "google_genai", "google.genai",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # Silence the tokenizers fork warning too.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_env() -> None:
    """Load API keys from a local .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logging.getLogger("crafter_experiment").warning(
            "python-dotenv not installed - .env not loaded (pip install python-dotenv)."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Crafter LLM experiment.")
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="path to the experiment config (default: config.yaml)")
    ap.add_argument("--no-live", action="store_true",
                    help="disable the real-time browser view (on by default)")
    ap.add_argument("--no-browser", action="store_true",
                    help="run the live view but don't auto-open a browser tab")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port for the live view (default: {DEFAULT_PORT})")
    ap.add_argument("--skip-run", action="store_true", help="don't run trials")
    ap.add_argument("--skip-analyze", action="store_true", help="don't build plots")
    ap.add_argument("--skip-viewer", action="store_true", help="don't build viewer.html")
    args = ap.parse_args()

    configure_logging()
    load_env()
    cfg = load_config(args.config)

    live = not args.no_live
    runner = None
    if not args.skip_run:
        runner = ExperimentRunner(
            cfg, live=live, live_port=args.port, open_browser=not args.no_browser
        )
        runner.run()

    if not args.skip_analyze:
        import analyze_results
        results = json.loads(cfg.results_path.read_text())
        rows = analyze_results.summarise(results)
        name = analyze_results.get_experiment_name(results, cfg.results_path)
        cfg.plots_dir.mkdir(parents=True, exist_ok=True)
        analyze_results.plot_success_rate(rows, cfg.plots_dir / "success_rate.png", name)
        analyze_results.plot_turns_to_success(rows, cfg.plots_dir / "turns_to_success.png", name)
        analyze_results.plot_think_time(rows, cfg.plots_dir / "think_time.png", name)
        analyze_results.plot_success_matrix(rows, cfg.plots_dir / "success_matrix.png", name)
        analyze_results.plot_tokens_vs_turns(results, cfg.plots_dir / "token_usage.png", name)
        analyze_results.print_summary(results, rows)

    if not args.skip_viewer:
        from viewer import build_viewer
        out = build_viewer(cfg.results_path, cfg.run_dir / "viewer.html")
        logging.getLogger("crafter_experiment").info("Replay viewer: %s", out)

    # Keep the live server up so the final state stays on screen until you quit.
    if live and runner is not None and runner.live is not None:
        runner.live.serve_until_interrupt()


if __name__ == "__main__":
    main()
