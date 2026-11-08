"""Short causal depthwise conv1d (kernel 4) fused with SiLU activation and gate branch multiplication."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ssm_short_conv_gate',
    'batch': 8,
    'seq_len': 8192,
    'd_inner': 4096,
    'kernel_size': 4,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    K = CONFIG['kernel_size']
    x = jax.random.normal(k1, (B, L, D), dtype=dtype)
    z = jax.random.normal(k2, (B, L, D), dtype=dtype)
    conv_weight = jax.random.normal(k3, (K, 1, D), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k4, (D,), dtype=dtype) * 0.02
    return x, z, conv_weight, conv_bias

def workload(x, z, conv_weight, conv_bias):
    orig_dtype = x.dtype
    K = CONFIG['kernel_size']
    D = CONFIG['d_inner']
    x_f32 = x.astype(jnp.float32)
    z_f32 = z.astype(jnp.float32)
    w_f32 = conv_weight.astype(jnp.float32)
    b_f32 = conv_bias.astype(jnp.float32)
    conv_out = lax.conv_general_dilated(
        lhs=x_f32,
        rhs=w_f32,
        window_strides=(1,),
        padding=((K - 1, 0),),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=D,
    )
    conv_out = conv_out + b_f32
    out = jax.nn.silu(conv_out) * jax.nn.silu(z_f32)
    return out.astype(orig_dtype)
