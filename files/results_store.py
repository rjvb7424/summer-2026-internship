"""Results persistence: one JSON per model, written after every trial."""

import json
from pathlib import Path

import config


class ResultsStore:
    """Crash-safe store. Rerunning main.py resumes where it stopped."""

    def __init__(self):
        self.directory = Path(config.RESULTS_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name):
        return self.directory / f"{model_name.replace('/', '__')}.json"

    def load_trials(self, model_name):
        """All completed trials for a model (empty list if none)."""
        path = self._path(model_name)
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def completed_trials(self, model_name):
        return len(self.load_trials(model_name))

    def append_trial(self, model_name, trial_record):
        """Append one trial and write to disk immediately."""
        trials = self.load_trials(model_name)
        trials.append(trial_record)
        self._path(model_name).write_text(json.dumps(trials, indent=2))

    def all_results(self):
        """model_name -> list of trial records, for every stored model."""
        results = {}
        for path in sorted(self.directory.glob("*.json")):
            results[path.stem.replace("__", "/")] = json.loads(path.read_text())
        return results
