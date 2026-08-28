from __future__ import annotations

import jax
import jax.numpy as jp
from flax import linen as nn


class ActorCritic(nn.Module):
    act_n: int = 8
    hidden: int = 64

    @nn.compact
    def __call__(self, obs: jax.Array):
        x = nn.tanh(nn.Dense(self.hidden)(obs))
        h1 = x
        x = nn.tanh(nn.Dense(self.hidden)(x))
        h2 = x
        mean = nn.tanh(nn.Dense(self.act_n)(x))
        log_std = self.param("log_std", nn.initializers.constant(-0.4), (self.act_n,))
        value = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value, h1, h2


def init_params(rng: jax.Array, obs_size: int, act_n: int = 8) -> dict:
    model = ActorCritic(act_n=act_n)
    dummy = jp.zeros((obs_size,), dtype=jp.float32)
    return model.init(rng, dummy)


def apply(params, obs: jax.Array, act_n: int = 8):
    return ActorCritic(act_n=act_n).apply(params, obs)
