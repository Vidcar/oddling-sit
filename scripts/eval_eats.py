"""Headless eat-count for stupid (random) or trained policy.

Usage (from Isaac Lab launcher):
  isaaclab.bat -p <this> --policy stupid --num_envs 16 --max_steps 400 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher, add_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=("stupid", "trained"), default="stupid")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--checkpoint", type=str, default=None)
add_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import oddling_lab.tasks.live  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from oddling.lab import TASK, trained_checkpoint


def main() -> int:
    env_cfg = parse_env_cfg(TASK, device=getattr(args_cli, "device", "cuda:0"), num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    unwrapped = env.unwrapped
    obs, _ = env.reset()
    for _ in range(args_cli.max_steps):
        act = 2.0 * torch.rand(unwrapped.num_envs, unwrapped.cfg.action_space, device=unwrapped.device) - 1.0
        env.step(act)
    eats = int(unwrapped.eats.float().mean().item())
    total = int(unwrapped.eats.sum().item())
    print(f"eats_mean={unwrapped.eats.float().mean().item():.3f} eats_sum={total} policy={args_cli.policy}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    simulation_app.close()
