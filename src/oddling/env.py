from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from oddling.lessons import Lesson

XML_PATH = Path(__file__).with_name("body.xml")
START_ENERGY = 12.0
DRAIN = 0.04
EAT_RADIUS = 0.45
EAT_ENERGY = 4.0
FOOD_HOME = jp.array([2.2, 0.0, 0.12], dtype=jp.float32)
ACT_N = 8


def load_mj_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


def torso_up(quat: jax.Array) -> jax.Array:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jp.array(
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ]
    )


class Field:
    def __init__(self) -> None:
        self.mj_model = load_mj_model()
        self.mjx_model = mjx.put_model(self.mj_model)
        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nu = int(self.mj_model.nu)
        self.torso_id = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"))
        self.mouth_id = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "mouth"))
        self.obs_size = self.nv + self.nu + 3 + 3 + 3 + 1

    def reset_one(self, rng: jax.Array) -> dict:
        data = mjx.make_data(self.mjx_model)
        qpos = data.qpos.at[:].set(self.mjx_model.qpos0)
        jitter = jax.random.uniform(rng, (self.nu,), minval=-0.12, maxval=0.12)
        qpos = qpos.at[7 : 7 + self.nu].add(jitter)
        data = data.replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)
        return {
            "data": data,
            "energy": jp.array(START_ENERGY, dtype=jp.float32),
            "eats": jp.array(0, dtype=jp.int32),
            "alive": jp.array(True),
            "ep_return": jp.array(0.0, dtype=jp.float32),
            "food": FOOD_HOME,
        }

    def reset(self, rng: jax.Array, n: int) -> dict:
        keys = jax.random.split(rng, n)
        return jax.vmap(self.reset_one)(keys)

    def observe_one(self, st: dict) -> jax.Array:
        data = st["data"]
        quat = data.qpos[3:7]
        up = torso_up(quat)
        torso = data.xpos[self.torso_id]
        rel = st["food"] - torso
        e = st["energy"] / START_ENERGY
        return jp.concatenate(
            [data.qvel, data.ctrl, up, rel, torso, jp.array([e], dtype=jp.float32)]
        )

    def observe(self, st: dict) -> jax.Array:
        return jax.vmap(self.observe_one)(st)

    def reward_one(self, st: dict, lesson: jax.Array) -> jax.Array:
        data = st["data"]
        quat = data.qpos[3:7]
        up = torso_up(quat)
        upright = jp.clip(up[2], 0.0, 1.0)
        move = jp.clip(data.qvel[0], -1.0, 2.0)
        survive = jp.where(st["alive"], 1.0, 0.0) + 0.35 * st["eats"].astype(jp.float32)
        return jp.dot(lesson, jp.array([survive, move, upright]))

    def step_one(self, st: dict, action: jax.Array, lesson: jax.Array, rng: jax.Array) -> tuple[dict, jax.Array, jax.Array]:
        action = jp.clip(action, -1.0, 1.0)
        data = mjx.step(self.mjx_model, st["data"].replace(ctrl=action))
        mouth = data.site_xpos[self.mouth_id]
        dist = jp.linalg.norm(mouth - st["food"])
        ate = dist < EAT_RADIUS
        energy = st["energy"] - DRAIN + ate.astype(jp.float32) * EAT_ENERGY
        eats = st["eats"] + ate.astype(jp.int32)
        k1, k2 = jax.random.split(rng)
        jitter = jax.random.uniform(k1, (3,), minval=jp.array([0.0, -0.5, 0.0]), maxval=jp.array([0.8, 0.5, 0.0]))
        food = jp.where(ate, FOOD_HOME + jitter, st["food"])
        height = data.xpos[self.torso_id][2]
        alive = (energy > 0.0) & (height > 0.12) & jp.isfinite(height)
        nxt = {
            "data": data,
            "energy": energy,
            "eats": eats,
            "alive": alive,
            "ep_return": st["ep_return"],
            "food": food,
        }
        r = self.reward_one(nxt, lesson)
        nxt["ep_return"] = st["ep_return"] + r
        reset_st = self.reset_one(k2)
        dead = ~alive

        def pick(a, b):
            return jax.tree.map(lambda x, y: jp.where(dead, x, y), a, b)

        out = pick(reset_st, nxt)
        return out, r, dead

    def step(self, st: dict, action: jax.Array, lesson: jax.Array, rng: jax.Array) -> tuple[dict, jax.Array, jax.Array]:
        n = action.shape[0]
        keys = jax.random.split(rng, n)
        les = jp.broadcast_to(lesson, (n, 3))
        return jax.vmap(self.step_one, in_axes=(0, 0, 0, 0))(st, action, les, keys)


def lesson_vec(lesson: Lesson) -> jax.Array:
    return jp.array([lesson.survive, lesson.move, lesson.upright], dtype=jp.float32)
