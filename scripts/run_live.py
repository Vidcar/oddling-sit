"""Launch Oddling-Live-Direct in Isaac Lab. Imports the gym task first."""

from __future__ import annotations

import argparse
import sys

import oddling_lab.tasks.live  # noqa: F401  — registers Oddling-Live-Direct

from isaaclab_rl.entrypoints import run_play_cli, run_random_agent_cli
from oddling.lab import TASK, trained_checkpoint


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--policy", choices=("stupid", "trained"), default="stupid")
    args, rest = p.parse_known_args(argv)
    rest = ["--task", TASK, *rest]
    if args.policy == "stupid":
        return run_random_agent_cli(rest) or 0
    ckpt = None
    if "--checkpoint" not in rest:
        found = trained_checkpoint()
        if found is None:
            print("No trained body yet. Run: oddling train", file=sys.stderr)
            return 2
        rest.extend(["--checkpoint", str(found)])
    return run_play_cli(["--rl_library", "rsl_rl", *rest]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
