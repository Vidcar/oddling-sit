from __future__ import annotations

import jax
import jax.numpy as jp
import optax

from oddling.env import FOOD_HOME, Field, lesson_vec
from oddling.learn import gae, gaussian_logprob, ppo_update, sample_action
from oddling.lessons import Lesson
from oddling.policy import apply as policy_apply
from oddling.policy import init_params


class Trainer:
    def __init__(self, n_envs: int = 128, n_steps: int = 16, seed: int = 0) -> None:
        self.field = Field()
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.rng = jax.random.PRNGKey(seed)
        self.rng, k = jax.random.split(self.rng)
        self.params = init_params(k, self.field.obs_size, self.field.nu)
        self.opt = optax.adam(3e-4)
        self.opt_state = self.opt.init(self.params)
        self.rng, k = jax.random.split(self.rng)
        self.state = self.field.reset(k, n_envs)
        self.lesson = Lesson.default()
        self.steps = 0
        self.chart: list[float] = []
        self._step = jax.jit(self.field.step)
        self._obs = jax.jit(self.field.observe)

    def set_lesson(self, lesson: Lesson) -> None:
        self.lesson = lesson

    def _rollout(self):
        obs_l, act_l, logp_l, val_l, rew_l, done_l = [], [], [], [], [], []
        les = lesson_vec(self.lesson)
        for _ in range(self.n_steps):
            obs = self._obs(self.state)
            mean, log_std, value, _, _ = policy_apply(self.params, obs)
            self.rng, k = jax.random.split(self.rng)
            act = sample_action(k, mean, log_std)
            logp = gaussian_logprob(mean, log_std, act)
            self.rng, k = jax.random.split(self.rng)
            self.state, rew, done = self._step(self.state, act, les, k)
            obs_l.append(obs)
            act_l.append(act)
            logp_l.append(logp)
            val_l.append(value)
            rew_l.append(rew)
            done_l.append(done)
            self.steps += self.n_envs
        last_obs = self._obs(self.state)
        _, _, last_v, _, _ = policy_apply(self.params, last_obs)
        rewards = jp.stack(rew_l)
        values = jp.concatenate([jp.stack(val_l), last_v[None, ...]], axis=0)
        dones = jp.stack(done_l)
        adv, ret = gae(rewards, values, dones)
        return (
            jp.concatenate(obs_l),
            jp.concatenate(act_l),
            jp.concatenate(logp_l),
            adv.reshape(-1),
            ret.reshape(-1),
            float(rewards.mean()),
        )

    def tick(self, unrolls: int = 1) -> float:
        last = 0.0
        for _ in range(max(1, unrolls)):
            obs, act, logp, adv, ret, mean_r = self._rollout()
            self.params, self.opt_state, _ = ppo_update(
                self.params, self.opt_state, self.opt, obs, act, logp, adv, ret
            )
            last = mean_r
            self.chart.append(mean_r)
            if len(self.chart) > 400:
                self.chart = self.chart[-400:]
        return last

    def best_index(self) -> int:
        ret = self.state["ep_return"]
        return int(jp.argmax(ret))

    def snapshot(self, i: int | None = None) -> dict:
        if i is None:
            i = self.best_index()
        data = jax.tree.map(lambda x: x[i], self.state["data"])
        return {
            "qpos": jax.device_get(data.qpos),
            "qvel": jax.device_get(data.qvel),
            "energy": float(self.state["energy"][i]),
            "eats": int(self.state["eats"][i]),
            "alive": bool(self.state["alive"][i]),
            "food": jax.device_get(self.state["food"][i]),
            "ep_return": float(self.state["ep_return"][i]),
        }

    def drop_food(self, pos=None) -> None:
        i = self.best_index()
        food = self.state["food"]
        if pos is None:
            pos = FOOD_NEAR
        food = food.at[i].set(jp.array(pos, dtype=jp.float32))
        self.state = {**self.state, "food": food}

    def brain(self) -> dict:
        obs = self._obs(self.state)
        i = self.best_index()
        mean, _, _, h1, h2 = policy_apply(self.params, obs)
        return {
            "h1": jax.device_get(h1[i]),
            "h2": jax.device_get(h2[i]),
            "mean": jax.device_get(mean[i]),
        }


FOOD_NEAR = FOOD_HOME
