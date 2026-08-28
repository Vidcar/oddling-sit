# GPU Sitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Oddling PRD r2 — sit in a 3D GPU field with a stupid default body, turbo, latest best, chart.

**Architecture:** MuJoCo MJX batched physics on JAX. PPO updates in-process. One pygame window shows the latest best (CPU MuJoCo copy of the best env) plus chart, brain, lessons. Turbo runs more PPO unrolls per frame. Same physics for learn and watch.

**Tech Stack:** Python 3.12, MuJoCo 3.12, mujoco-mjx, JAX, Flax, Optax, pygame. GPU via jax[cuda12] when present.

## Global Constraints

- Product is docs/prd.md r2 approved 2026-08-28.
- Not the Godot Oddling garden. No Train, no overnight job, no Classroom, no snap-kit this sitting.
- Default body starts random. Chart/number required. Plateau allowed.
- Local Windows, capable NVIDIA GPU. RTX 3090 is available on this machine.
- Verify: `python -m pytest tests -q` then `oddling` lands in the sitting.

## Tasks

1. Repo + PRD + README sitting guide
2. MJX field env (energy, food, death reset, lessons)
3. In-process PPO trainer + chart history
4. pygame sitting (view, turbo, drop food, brain, persist)
5. pytest green on CPU; GPU used when jax sees cuda
