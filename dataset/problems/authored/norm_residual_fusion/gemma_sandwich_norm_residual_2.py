"""Gemma sandwich norm: post-branch RMSNorm(1+w) -> residual add -> pre-norm RMSNorm(1+w), returning (normed, residual)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma_sandwich_norm_residual',
    'num_tokens': 16384,
    'hidden_size': 3584,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_h, k_r = jax.random.split(key, 2)
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_size']
    branch = jax.random.normal(k_h, (T, D), dtype=jnp.float32).astype(dtype)
    residual = jax.random.normal(k_r, (T, D), dtype=jnp.float32).astype(dtype)
    w_post = jnp.zeros((D,), dtype=jnp.float32)
    w_pre = jnp.zeros((D,), dtype=jnp.float32)
    return branch, residual, w_post, w_pre


def _gemma_rmsnorm_f32(x, w, eps):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * (1.0 + w.astype(jnp.float32))


def workload(branch, residual, w_post, w_pre):
    eps = CONFIG['eps']
    out_dtype = residual.dtype
    h = branch.astype(jnp.float32)
    r = residual.astype(jnp.float32)
    post = _gemma_rmsnorm_f32(h, w_post, eps)
    r_new = r + post
    y = _gemma_rmsnorm_f32(r_new, w_pre, eps)
    return y.astype(out_dtype), r_new.astype(out_dtype)
