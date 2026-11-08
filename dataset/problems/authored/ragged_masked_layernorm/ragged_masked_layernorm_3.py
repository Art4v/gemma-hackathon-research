"""Ragged masked LayerNorm over padded variable-length sequences that zeroes padded positions."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_masked_layernorm',
    'batch_size': 32,
    'max_seq_len': 2048,
    'hidden_size': 4096,
    'min_seq_len': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_len = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    S = CONFIG['max_seq_len']
    H = CONFIG['hidden_size']
    min_S = CONFIG['min_seq_len']

    x = jax.random.normal(k_x, (B, S, H), dtype=dtype)
    seq_lens = jax.random.randint(k_len, (B,), minval=min_S, maxval=S + 1, dtype=jnp.int32)
    weight = jnp.ones((H,), dtype=dtype)
    bias = jnp.zeros((H,), dtype=dtype)
    return x, seq_lens, weight, bias


def workload(x, seq_lens, weight, bias):
    B, S, H = x.shape
    eps = CONFIG['eps']

    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
    x_norm = (x_f32 - mean) * jax.lax.rsqrt(var + eps)

    out_f32 = x_norm * weight.astype(jnp.float32) + bias.astype(jnp.float32)

    positions = jnp.arange(S, dtype=jnp.int32)[None, :, None]
    mask = positions < seq_lens[:, None, None]
    out_masked = jnp.where(mask, out_f32, 0.0)
    return out_masked.astype(x.dtype)
