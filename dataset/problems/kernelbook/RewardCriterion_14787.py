"""[bf16 variant] RewardCriterion: gather log-probs at seq, shifted (seq > pad) mask, masked negative reward-weighted mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RewardCriterion_14787',
    'batch': 512,
    'seq_len': 128,
    'vocab': 1024,
    'pad_id': 0,
    'seq_min': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_input, k_seq, k_reward = jax.random.split(jax.random.key(0), 3)
    n, l, v = CONFIG['batch'], CONFIG['seq_len'], CONFIG['vocab']
    log_probs = jax.random.normal(k_input, (n, l, v), dtype)
    seq = jax.random.randint(k_seq, (n, l), CONFIG['seq_min'], v, jnp.int32)
    reward = jax.random.uniform(k_reward, (n, l), dtype)
    return log_probs, seq, reward


def workload(log_probs, seq, reward):
    picked = jnp.take_along_axis(log_probs, seq[:, :, None], axis=2).squeeze(2)
    picked = picked.reshape(-1)
    flat_reward = reward.reshape(-1)
    mask = (seq > CONFIG['pad_id']).astype(reward.dtype)
    mask = jnp.concatenate(
        [jnp.ones((mask.shape[0], 1), mask.dtype), mask[:, :-1]], axis=1
    ).reshape(-1)
    output = -picked * flat_reward * mask
    return jnp.sum(output) / jnp.sum(mask)
