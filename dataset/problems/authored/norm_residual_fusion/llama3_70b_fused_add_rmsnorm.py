"""Fused residual addition and RMSNorm returning normed output and updated residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_70b_fused_add_rmsnorm',
    'batch_size': 4,
    'seq_len': 4096,
    'hidden_dim': 8192,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    shape = (CONFIG['batch_size'], CONFIG['seq_len'], CONFIG['hidden_dim'])
    x = jax.random.normal(k1, shape, dtype=dtype)
    residual = jax.random.normal(k2, shape, dtype=dtype)
    weight = jnp.ones((CONFIG['hidden_dim'],), dtype=dtype)
    return x, residual, weight


def workload(x, residual, weight):
    eps = CONFIG['eps']
    residual_out = (x.astype(jnp.float32) + residual.astype(jnp.float32)).astype(x.dtype)
    r_f32 = residual_out.astype(jnp.float32)
    variance = jnp.mean(jnp.square(r_f32), axis=-1, keepdims=True)
    normed = r_f32 * jax.lax.rsqrt(variance + eps)
    out = (normed * weight.astype(jnp.float32)).astype(x.dtype)
    return out, residual_out
