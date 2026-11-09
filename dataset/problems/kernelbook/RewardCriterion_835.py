"""[bf16 variant] RewardCriterion: shifted (seq > 0) mask, masked -input*reward sum, normalized by mask sum."""
import jax
import jax.numpy as jnp

CONFIG = {'name': 'RewardCriterion_835', 'batch': 4096, 'seq_len': 4096}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['seq_len'])
    inp = jax.random.uniform(k1, shape, dtype)
    seq = jax.random.uniform(k2, shape, dtype)
    reward = jax.random.uniform(k3, shape, dtype)
    return inp, seq, reward


def workload(inp, seq, reward):
    mask = (seq > 0).astype(inp.dtype)
    mask = jnp.concatenate(
        [jnp.ones((mask.shape[0], 1), inp.dtype), mask[:, :-1]], axis=1)
    output = -inp * reward * mask
    return jnp.sum(output) / jnp.sum(mask)
