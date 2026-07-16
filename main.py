"""
main.py
=======

One command to run an experiment end to end:

    python main.py                 # run + plots + viewer, using config.yaml
    python main.py my_config.yaml
    python main.py --live          # watch it in real time in your browser
    python main.py --skip-analyze  # just run the trials
    python main.py --skip-run      # only (re)build plots + viewer from results

The experiment is defined entirely by the YAML config; this file only wires the
pieces together and configures logging.
"""

from __future__ import annotations

import argparse
import json
import logging

from config import load_config
from experiment import ExperimentRunner
from live_viewer import DEFAULT_PORT


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Crafter LLM experiment.")
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="path to the experiment config (default: config.yaml)")
    ap.add_argument("--live", action="store_true",
                    help="serve a real-time browser view while the run happens")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port for the live view (default: {DEFAULT_PORT})")
    ap.add_argument("--skip-run", action="store_true", help="don't run trials")
    ap.add_argument("--skip-analyze", action="store_true", help="don't build plots")
    ap.add_argument("--skip-viewer", action="store_true", help="don't build viewer.html")
    args = ap.parse_args()

    configure_logging()
    cfg = load_config(args.config)

    runner = None
    if not args.skip_run:
        runner = ExperimentRunner(cfg, live=args.live, live_port=args.port)
        runner.run()

    if not args.skip_analyze:
        import analyze_results
        results = json.loads(cfg.results_path.read_text())
        rows = analyze_results.summarise(results)
        cfg.plots_dir.mkdir(parents=True, exist_ok=True)
        analyze_results.plot_success_rate(rows, cfg.plots_dir / "success_rate.png")
        analyze_results.plot_turns_to_success(rows, cfg.plots_dir / "turns_to_success.png")
        analyze_results.plot_think_time(rows, cfg.plots_dir / "think_time.png")
        analyze_results.plot_success_matrix(rows, cfg.plots_dir / "success_matrix.png")
        analyze_results.print_summary(results, rows)

    if not args.skip_viewer:
        from viewer import build_viewer
        out = build_viewer(cfg.results_path)
        logging.getLogger("crafter_experiment").info("Viewer: %s", out)

    # Keep the live server up so the final state stays on screen until you quit.
    if args.live and runner is not None and runner.live is not None:
        runner.live.serve_until_interrupt()


if __name__ == "__main__":
    main()
