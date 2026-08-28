from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jp

from oddling.lessons import Lesson
from oddling.trainer import Trainer


def test_trainer_tick_moves_chart() -> None:
    tr = Trainer(n_envs=4, n_steps=4, seed=1)
    assert tr.chart == []
    tr.tick(unrolls=2)
    assert len(tr.chart) == 2
    assert all(isinstance(x, float) for x in tr.chart)
    snap = tr.snapshot()
    assert snap["food"].shape == (3,)
    assert "energy" in snap
    tr.set_lesson(Lesson(survive=1.0, move=1.0, upright=0.2))
    tr.drop_food(jp.array([1.0, 0.0, 0.12]))
    brain = tr.brain()
    assert "h1" in brain and "mean" in brain
