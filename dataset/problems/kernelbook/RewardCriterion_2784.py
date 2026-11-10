"""[bf16 variant] RewardCriterion: gather log-probs by seq, shifted pad mask, negative reward-weighted masked mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RewardCriterion_2784',
    'batch': 128,
    'seq_len': 256,
    'vocab': 1024,
    'pad_period': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_input, k_seq, k_pad, k_reward = jax.random.split(key, 4)
    batch, seq_len, vocab = CONFIG['batch'], CONFIG['seq_len'], CONFIG['vocab']
    log_probs = jax.random.uniform(k_input, (batch, seq_len, vocab), dtype)
    tokens = jax.random.randint(k_seq, (batch, seq_len), 0, vocab, jnp.int32)
    pad = jax.random.randint(k_pad, (batch, seq_len), 0, CONFIG['pad_period'], jnp.int32)
    seq = jnp.where(pad == 0, jnp.zeros_like(tokens), tokens)
    reward = jax.random.uniform(k_reward, (batch, seq_len), dtype)
    return log_probs, seq, reward


def workload(log_probs, seq, reward):
    picked = jnp.take_along_axis(log_probs, seq[:, :, None], axis=2)
    picked = jnp.squeeze(picked, axis=2).reshape(-1)
    reward = reward.reshape(-1)
    mask = seq > 0
    mask = jnp.concatenate([jnp.ones_like(mask[:, :1]), mask[:, :-1]], axis=1).reshape(-1)
    output = -picked * reward * mask
    return jnp.sum(output) / jnp.sum(mask)
