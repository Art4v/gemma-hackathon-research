"""Gemma-2 27B fused post-RMSNorm, residual add, and pre-RMSNorm with unit-offset weights."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_27b_post_norm_residual_pre_norm',
    'batch_size': 8,
    'seq_len': 2048,
    'hidden_size': 4608,
    'eps': 1e-6,
    'weight_offset': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shape = (CONFIG['batch_size'], CONFIG['seq_len'], CONFIG['hidden_size'])
    w_shape = (CONFIG['hidden_size'],)
    residual = jax.random.normal(k1, shape, dtype=dtype)
    sublayer_out = jax.random.normal(k2, shape, dtype=dtype)
    post_norm_weight = jax.random.normal(k3, w_shape, dtype=dtype) * 0.02
    pre_norm_weight = jax.random.normal(k4, w_shape, dtype=dtype) * 0.02
    return residual, sublayer_out, post_norm_weight, pre_norm_weight


def _gemma_rmsnorm(x, weight, eps, offset):
    x_f32 = x.astype(jnp.float32)
    w_f32 = weight.astype(jnp.float32) + offset
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + eps)
    return (normed * w_f32).astype(x.dtype)


def workload(residual, sublayer_out, post_norm_weight, pre_norm_weight):
    eps = CONFIG['eps']
    offset = CONFIG['weight_offset']
    post_normed = _gemma_rmsnorm(sublayer_out, post_norm_weight, eps, offset)
    updated_residual = (residual.astype(jnp.float32) + post_normed.astype(jnp.float32)).astype(residual.dtype)
    pre_normed = _gemma_rmsnorm(updated_residual, pre_norm_weight, eps, offset)
    return pre_normed, updated_residual
