from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from oddling.lab import BAT, USER_TRAINED, trained_checkpoint

ROOT = Path(__file__).resolve().parents[2]
RUN_LIVE = ROOT / "scripts" / "run_live.py"
TRAIN_LIVE = ROOT / "scripts" / "train_live.py"


def _run(script: Path, extra: list[str]) -> int:
    if not BAT.is_file():
        print(f"Isaac Lab not found at {BAT}. Set ISAACLAB_PATH.", file=sys.stderr)
        return 2
    cmd = [str(BAT), "-p", str(script), *extra]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def cmd_watch(kind: str) -> int:
    extra = ["--policy", kind, "--num_envs", "1"]
    if kind == "trained" and trained_checkpoint() is None:
        print("No trained body yet. Run: oddling train", file=sys.stderr)
        return 2
    return _run(RUN_LIVE, extra)


def cmd_train() -> int:
    USER_TRAINED.mkdir(parents=True, exist_ok=True)
    rc = _run(TRAIN_LIVE, ["--headless", "--max_iterations", "1000"])
    if rc != 0:
        return rc
    ckpt = trained_checkpoint()
    if ckpt is not None and ckpt.parent != USER_TRAINED:
        dest = USER_TRAINED / ckpt.name
        dest.write_bytes(ckpt.read_bytes())
        print(f"saved {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oddling", description="Watch a four-leg critter try to live.")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="watch stupid or trained")
    w.add_argument("kind", choices=("stupid", "trained"))
    sub.add_parser("train", help="train the body to live")
    args = p.parse_args(argv)
    if args.cmd == "watch":
        return cmd_watch(args.kind)
    if args.cmd == "train":
        return cmd_train()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
