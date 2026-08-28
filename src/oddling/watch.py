from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from oddling.lab import BAT, PLAY_PY, RANDOM_PY, TASK, TRAIN_PY, USER_TRAINED, trained_checkpoint


def _run(script: Path, extra: list[str]) -> int:
    if not BAT.is_file():
        print(f"Isaac Lab not found at {BAT}. Set ISAACLAB_PATH.", file=sys.stderr)
        return 2
    cmd = [str(BAT), "-p", str(script), "--task", TASK, *extra]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def cmd_watch(kind: str) -> int:
    if kind == "stupid":
        return _run(RANDOM_PY, ["--num_envs", "1"])
    ckpt = trained_checkpoint()
    if ckpt is None:
        print("No trained body yet. Run: oddling train", file=sys.stderr)
        return 2
    return _run(PLAY_PY, ["--num_envs", "1", "--checkpoint", str(ckpt)])


def cmd_train() -> int:
    USER_TRAINED.mkdir(parents=True, exist_ok=True)
    rc = _run(TRAIN_PY, ["--headless", "--max_iterations", "1000"])
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
