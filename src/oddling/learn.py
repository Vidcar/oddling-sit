from __future__ import annotations

import jax
import jax.numpy as jp
import optax

from oddling.policy import apply as policy_apply


def gaussian_logprob(mean, log_std, action):
    std = jp.exp(log_std)
    return -0.5 * (((action - mean) / (std + 1e-6)) ** 2 + 2 * log_std + jp.log(2 * jp.pi)).sum(-1)


def sample_action(rng, mean, log_std):
    return mean + jax.random.normal(rng, mean.shape) * jp.exp(log_std)


def ppo_update(params, opt_state, opt, obs, actions, old_logp, adv, ret, clip=0.2):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    def loss_fn(p):
        mean, log_std, value, _, _ = policy_apply(p, obs)
        logp = gaussian_logprob(mean, log_std, actions)
        ratio = jp.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = jp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
        policy_loss = -jp.minimum(surr1, surr2).mean()
        value_loss = 0.5 * ((value - ret) ** 2).mean()
        entropy = (0.5 + log_std).mean()
        return policy_loss + 0.5 * value_loss - 0.01 * entropy

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    def body(adv, inp):
        r, v, n_v, d = inp
        mask = 1.0 - d.astype(jp.float32)
        delta = r + gamma * n_v * mask - v
        adv = delta + gamma * lam * mask * adv
        return adv, adv

    _, advs = jax.lax.scan(
        body,
        jp.zeros(rewards.shape[1], dtype=jp.float32),
        (rewards, values[:-1], values[1:], dones),
        reverse=True,
    )
    returns = advs + values[:-1]
    return advs, returns
