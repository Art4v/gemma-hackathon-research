"""Fused Gated Group RMSNorm as used in Mamba-2 and Gated DeltaNet."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mamba2_gated_rmsnorm',
    'batch': 8,
    'seq_len': 4096,
    'num_groups': 8,
    'group_dim': 512,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    B = CONFIG['batch']
    T = CONFIG['seq_len']
    G = CONFIG['num_groups']
    D = CONFIG['group_dim']
    x = jax.random.normal(k1, (B, T, G, D), dtype=dtype)
    gate = jax.random.normal(k2, (B, T, G, D), dtype=dtype)
    weight = jnp.ones((G, D), dtype=dtype)
    return x, gate, weight


def workload(x, gate, weight):
    eps = CONFIG['eps']
    x_f32 = x.astype(jnp.float32)
    gate_f32 = gate.astype(jnp.float32)
    weight_f32 = weight.astype(jnp.float32)

    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * lax.rsqrt(variance + eps)
    gated = x_norm * jax.nn.silu(gate_f32)
    out = gated * weight_f32
    return out.astype(x.dtype)
