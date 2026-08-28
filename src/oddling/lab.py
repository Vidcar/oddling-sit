from __future__ import annotations

import os
from pathlib import Path

ISAACLAB = Path(os.environ.get("ISAACLAB_PATH", r"D:\codeprojects\IsaacLab"))
TASK = "Oddling-Live-Direct"
PYTHON = ISAACLAB / "env_isaaclab" / "Scripts" / "python.exe"
BAT = ISAACLAB / "isaaclab.bat"
TRAIN_PY = ISAACLAB / "scripts" / "reinforcement_learning" / "train.py"
PLAY_PY = ISAACLAB / "scripts" / "reinforcement_learning" / "play.py"
RANDOM_PY = ISAACLAB / "scripts" / "environments" / "random_agent.py"
USER_TRAINED = Path.home() / ".oddling" / "trained"
REPO_TRAINED = Path(__file__).resolve().parents[2] / "runs" / "trained"


def trained_checkpoint() -> Path | None:
    for folder in (USER_TRAINED, REPO_TRAINED):
        if not folder.is_dir():
            continue
        pts = sorted(folder.glob("*.pt"))
        if pts:
            return pts[-1]
    logs = Path(__file__).resolve().parents[2] / "logs" / "rsl_rl" / "oddling_live"
    if not logs.is_dir():
        logs = ISAACLAB / "logs" / "rsl_rl" / "oddling_live"
    if logs.is_dir():
        runs = sorted(p for p in logs.iterdir() if p.is_dir())
        if runs:
            exported = runs[-1] / "exported"
            for name in ("policy.pt", "model.pt"):
                cand = exported / name
                if cand.is_file():
                    return cand
            models = sorted(runs[-1].glob("model_*.pt"))
            if models:
                return models[-1]
    return None

