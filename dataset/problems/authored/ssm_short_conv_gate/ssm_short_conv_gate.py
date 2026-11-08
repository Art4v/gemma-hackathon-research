"""Short causal depthwise conv1d (kernel 4) followed by SiLU and gate branch multiply."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ssm_short_conv_gate',
    'batch': 16,
    'seq_len': 4096,
    'd_inner': 4096,
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
    gate = jax.random.normal(k2, (B, L, D), dtype=dtype)
    conv_weight = jax.random.normal(k3, (D, 1, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k4, (D,), dtype=dtype) * 0.02
    return x, gate, conv_weight, conv_bias


def workload(x, gate, conv_weight, conv_bias):
    orig_dtype = x.dtype
    K = CONFIG['kernel_size']
    D = CONFIG['d_inner']

    x_f32 = x.astype(jnp.float32)
    gate_f32 = gate.astype(jnp.float32)
    w_f32 = conv_weight.astype(jnp.float32)
    b_f32 = conv_bias.astype(jnp.float32)

    # Transpose to (B, D, L) for causal depthwise conv1d
    x_bdl = jnp.swapaxes(x_f32, 1, 2)
    x_padded = jnp.pad(x_bdl, ((0, 0), (0, 0), (K - 1, 0)))

    conv_out = lax.conv_general_dilated(
        lhs=x_padded,
        rhs=w_f32,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=('NCH', 'OIH', 'NCH'),
        feature_group_count=D,
    )

    # Transpose back to (B, L, D) and add bias
    conv_out = jnp.swapaxes(conv_out, 1, 2) + b_f32

    # SiLU activation on conv output and gate branch, then elementwise multiply
    conv_act = jax.nn.silu(conv_out)
    gate_act = jax.nn.silu(gate_f32)
    out = conv_act * gate_act

    return out.astype(orig_dtype)
