"""Short causal depthwise conv1d (kernel 4) followed by SiLU and branch gating."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mamba_short_conv_silu_gate',
    'batch': 16,
    'seq_len': 8192,
    'd_inner': 2048,
    'kernel_size': 4,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    K = CONFIG['kernel_size']
    x = jax.random.normal(k1, (B, L, D), dtype=dtype)
    z = jax.random.normal(k2, (B, L, D), dtype=dtype)
    weight = jax.random.normal(k3, (K, 1, D), dtype=dtype) * 0.02
    bias = jax.random.normal(k4, (D,), dtype=dtype) * 0.02
    return x, z, weight, bias

def workload(x, z, weight, bias):
    K = CONFIG['kernel_size']
    D = CONFIG['d_inner']
    x_f32 = x.astype(jnp.float32)
    z_f32 = z.astype(jnp.float32)
    w_f32 = weight.astype(jnp.float32)
    b_f32 = bias.astype(jnp.float32)

    dn = lax.conv_dimension_numbers(x_f32.shape, w_f32.shape, ('NWC', 'WIO', 'NWC'))
    conv_out = lax.conv_general_dilated(
        x_f32,
        w_f32,
        window_strides=(1,),
        padding=((K - 1, 0),),
        dimension_numbers=dn,
        feature_group_count=D,
    )
    conv_out = conv_out + b_f32
    conv_act = jax.nn.silu(conv_out)
    gate_act = jax.nn.silu(z_f32)
    out = conv_act * gate_act
    return out.astype(x.dtype)
