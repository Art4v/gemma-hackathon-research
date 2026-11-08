"""Fused bias add, residual add, and LayerNorm returning normed output and updated residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gpt_neox_20b_bias_residual_layernorm',
    'batch_size': 4,
    'seq_len': 4096,
    'hidden_size': 6144,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    H = CONFIG['hidden_size']
    hidden_states = jax.random.normal(k1, (B, S, H), dtype=dtype)
    residual = jax.random.normal(k2, (B, S, H), dtype=dtype)
    bias = jax.random.normal(k3, (H,), dtype=dtype) * 0.02
    gamma = jnp.ones((H,), dtype=dtype)
    beta = jnp.zeros((H,), dtype=dtype)
    return hidden_states, residual, bias, gamma, beta


def workload(hidden_states, residual, bias, gamma, beta):
    eps = CONFIG['eps']
    in_dtype = hidden_states.dtype
    updated_residual = (hidden_states.astype(jnp.float32) + bias.astype(jnp.float32) + residual.astype(jnp.float32)).astype(in_dtype)
    x_f32 = updated_residual.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    centered = x_f32 - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    normed = centered * rstd
    out = normed * gamma.astype(jnp.float32) + beta.astype(jnp.float32)
    return out.astype(in_dtype), updated_residual
