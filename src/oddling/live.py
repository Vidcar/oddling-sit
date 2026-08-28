from __future__ import annotations

import math
from dataclasses import dataclass

START_ENERGY = 72.0
DRAIN = 0.02
EAT_RADIUS = 0.32
EAT_ENERGY = 12.0
FOOD_HOME = (1.05, 0.0, 0.12)
FOOD_CLEAR = 0.70
FOOD_JITTER_X = 0.35
FOOD_JITTER_Y = 0.35
EAT_LOCK = 30
COLLAPSE_STEPS = 24


def mouth_reaches(mouth_xyz: tuple[float, float, float], food_xyz: tuple[float, float, float]) -> bool:
    dx = mouth_xyz[0] - food_xyz[0]
    dy = mouth_xyz[1] - food_xyz[1]
    dz = mouth_xyz[2] - food_xyz[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5 < EAT_RADIUS


@dataclass(frozen=True)
class LiveState:
    energy: float
    eats: int
    food: tuple[float, float, float]
    collapsed: bool
    collapse_left: int
    rng: int = 1

    @classmethod
    def spawn(cls) -> LiveState:
        return cls(energy=START_ENERGY, eats=0, food=FOOD_HOME, collapsed=False, collapse_left=0, rng=1)


def _jitter(rng: int, mouth_xyz: tuple[float, float, float]) -> tuple[tuple[float, float, float], int]:
    """Place food in front of the mouth, always outside the eat bubble."""
    rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
    dist = FOOD_CLEAR + (rng % 1000) / 1000.0 * FOOD_JITTER_X
    rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
    ang = ((rng % 1000) / 1000.0 - 0.5) * 1.2
    x = mouth_xyz[0] + dist * math.cos(ang)
    y = mouth_xyz[1] + dist * math.sin(ang)
    return (x, y, FOOD_HOME[2]), rng


def step_live(
    st: LiveState,
    mouth_xyz: tuple[float, float, float],
    ate: bool | None = None,
) -> LiveState:
    if st.collapsed:
        left = st.collapse_left - 1
        if left <= 0:
            return LiveState(
                energy=START_ENERGY,
                eats=0,
                food=FOOD_HOME,
                collapsed=False,
                collapse_left=0,
                rng=st.rng,
            )
        return LiveState(st.energy, st.eats, st.food, True, left, st.rng)

    if ate is None:
        ate = mouth_reaches(mouth_xyz, st.food)
    energy = st.energy - DRAIN + (EAT_ENERGY if ate else 0.0)
    eats = st.eats + (1 if ate else 0)
    food, rng = st.food, st.rng
    if ate:
        food, rng = _jitter(st.rng, mouth_xyz)
    if energy <= 0.0:
        return LiveState(0.0, eats, food, True, COLLAPSE_STEPS, rng)
    return LiveState(energy, eats, food, False, 0, rng)
