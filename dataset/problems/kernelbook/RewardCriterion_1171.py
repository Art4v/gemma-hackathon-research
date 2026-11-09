"""RewardCriterion: gather(logprobs, seq) -> shifted (seq>0) mask -> sum(-logp * reward * mask) / sum(mask)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RewardCriterion_1171',
    'batch': 32,
    'seq_len': 256,
    'vocab': 4096,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, t, v = CONFIG['batch'], CONFIG['seq_len'], CONFIG['vocab']
    logprobs = jax.random.uniform(k1, (b, t, v), dtype)
    seq = jax.random.randint(k2, (b, t), 0, v, dtype=jnp.int32)
    reward = jax.random.uniform(k3, (b, t), dtype)
    return logprobs, seq, reward


def workload(logprobs, seq, reward):
    picked = jnp.take_along_axis(logprobs, seq[:, :, None], axis=2)[:, :, 0].reshape(-1)
    mask = (seq > 0).astype(logprobs.dtype)
    mask = jnp.concatenate(
        [jnp.ones((mask.shape[0], 1), mask.dtype), mask[:, :-1]], axis=1
    ).reshape(-1)
    out = -picked * reward.reshape(-1) * mask
    return jnp.sum(out) / jnp.sum(mask)
