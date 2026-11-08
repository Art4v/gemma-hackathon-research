"""Gated RMSNorm operator with SiLU gating and group-wise normalization from Mamba2 and Gated DeltaNet."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mamba2_gated_rmsnorm',
    'batch_size': 8,
    'seq_len': 4096,
    'num_groups': 8,
    'group_dim': 512,
    'eps': 1e-5,
    'norm_before_gate': False,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    shape = (
        CONFIG['batch_size'],
        CONFIG['seq_len'],
        CONFIG['num_groups'],
        CONFIG['group_dim'],
    )
    x = jax.random.normal(k1, shape, dtype=dtype)
    gate = jax.random.normal(k2, shape, dtype=dtype)
    weight = jnp.ones((CONFIG['num_groups'], CONFIG['group_dim']), dtype=dtype)
    return x, gate, weight


def workload(x, gate, weight):
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    gate_f32 = gate.astype(jnp.float32)
    weight_f32 = weight.astype(jnp.float32)

    silu_gate = jax.nn.silu(gate_f32)

    if CONFIG['norm_before_gate']:
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(variance + CONFIG['eps'])
        out = x_normed * weight_f32 * silu_gate
    else:
        gated_x = x_f32 * silu_gate
        variance = jnp.mean(jnp.square(gated_x), axis=-1, keepdims=True)
        x_normed = gated_x * jax.lax.rsqrt(variance + CONFIG['eps'])
        out = x_normed * weight_f32

    return out.astype(orig_dtype)
