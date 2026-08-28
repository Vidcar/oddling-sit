from __future__ import annotations

from oddling.live import (
    DRAIN,
    EAT_ENERGY,
    FOOD_CLEAR,
    FOOD_HOME,
    START_ENERGY,
    LiveState,
    mouth_reaches,
    step_live,
)


def test_energy_drains_without_eat() -> None:
    st = LiveState.spawn()
    nxt = step_live(st, mouth_xyz=FOOD_HOME, ate=False)
    assert nxt.energy == START_ENERGY - DRAIN
    assert nxt.eats == 0
    assert nxt.collapsed is False


def test_mouth_on_food_eats_and_food_moves_out_of_reach() -> None:
    st = LiveState.spawn()
    mouth = (0.22, 0.0, 0.35)
    nxt = step_live(st, mouth_xyz=mouth, ate=True)
    assert nxt.energy == START_ENERGY - DRAIN + EAT_ENERGY
    assert nxt.eats == 1
    assert nxt.food != FOOD_HOME
    assert not mouth_reaches(mouth, nxt.food)
    dx = nxt.food[0] - mouth[0]
    dy = nxt.food[1] - mouth[1]
    dist = (dx * dx + dy * dy) ** 0.5
    assert dist >= FOOD_CLEAR - 1e-5


def test_starve_collapses_then_same_run_continues() -> None:
    st = LiveState(energy=DRAIN * 0.5, eats=2, food=FOOD_HOME, collapsed=False, collapse_left=0)
    dead = step_live(st, mouth_xyz=(0.0, 0.0, 0.4), ate=False)
    assert dead.collapsed is True
    assert dead.collapse_left > 0
    while dead.collapse_left > 0:
        dead = step_live(dead, mouth_xyz=(0.0, 0.0, 0.05), ate=False)
    assert dead.collapsed is False
    assert dead.energy == START_ENERGY
    assert dead.food == FOOD_HOME
    assert dead.eats == 0
