"""Fused Gated RMSNorm: RMSNorm(x) * SiLU(gate) * weight from Mamba-2 and Gated DeltaNet."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mamba2_gated_rmsnorm',
    'batch': 8,
    'seq_len': 4096,
    'hidden_size': 4096,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    b, s, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_size']
    x = jax.random.normal(k1, (b, s, h), dtype=dtype)
    gate = jax.random.normal(k2, (b, s, h), dtype=dtype)
    weight = jnp.ones((h,), dtype=dtype)
    return x, gate, weight


def workload(x, gate, weight):
    eps = CONFIG['eps']
    x_f32 = x.astype(jnp.float32)
    gate_f32 = gate.astype(jnp.float32)
    weight_f32 = weight.astype(jnp.float32)

    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * jax.lax.rsqrt(variance + eps)
    gated = x_norm * jax.nn.silu(gate_f32) * weight_f32
    return gated.astype(x.dtype)
