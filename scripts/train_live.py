"""Train Oddling-Live-Direct. Imports the gym task first."""

from __future__ import annotations

import sys

import oddling_lab.tasks.live  # noqa: F401

from isaaclab_rl.entrypoints import run_train_cli
from oddling.lab import TASK


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--task" not in argv:
        argv = ["--task", TASK, *argv]
    if "--rl_library" not in argv:
        argv = ["--rl_library", "rsl_rl", *argv]
    return run_train_cli(argv) or 0


if __name__ == "__main__":
    raise SystemExit(main())
