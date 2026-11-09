"""RewardCriterion: shifted (seq > 0) mask, -sum(input * reward * mask) / sum(mask)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RewardCriterion_8829',
    'batch': 4096,
    'seq_len': 4096,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['seq_len'])
    k_in, k_seq, k_rew = jax.random.split(jax.random.key(0), 3)
    inp = jax.random.uniform(k_in, shape, dtype)
    seq = jax.random.uniform(k_seq, shape, dtype)
    reward = jax.random.uniform(k_rew, shape, dtype)
    return inp, seq, reward


def workload(inp, seq, reward):
    mask = (seq > 0).astype(inp.dtype)
    ones = jnp.ones((mask.shape[0], 1), dtype=inp.dtype)
    mask = jnp.concatenate([ones, mask[:, :-1]], axis=1)
    output = -inp * reward * mask
    return jnp.sum(output) / jnp.sum(mask)
