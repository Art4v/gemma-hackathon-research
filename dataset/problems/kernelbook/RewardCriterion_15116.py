"""[bf16 variant] RewardCriterion: shifted (seq>0) mask, -input*reward*mask summed and normalized by mask sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RewardCriterion_15116',
    'batch': 4096,
    'seq_len': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    k_in, k_seq, k_rew = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['seq_len'])
    inp = jax.random.uniform(k_in, shape, dtype)
    seq = jax.random.uniform(k_seq, shape, dtype)
    reward = jax.random.uniform(k_rew, shape, dtype)
    return inp, seq, reward


def workload(inp, seq, reward):
    mask = (seq > 0).astype(inp.dtype)
    # Shift right by one step: the first token always counts.
    ones = jnp.ones((mask.shape[0], 1), dtype=mask.dtype)
    mask = jnp.concatenate([ones, mask[:, :-1]], axis=1).reshape(-1)
    output = -inp.reshape(-1) * reward.reshape(-1) * mask
    return jnp.sum(output) / jnp.sum(mask)
