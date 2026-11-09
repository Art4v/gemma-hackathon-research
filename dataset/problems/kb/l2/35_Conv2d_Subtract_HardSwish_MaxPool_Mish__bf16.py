"""[bf16 variant] Conv2d followed by scalar subtraction, HardSwish, MaxPool2d, and Mish."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '35_Conv2d_Subtract_HardSwish_MaxPool_Mish',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'subtract_value': 0.5,
    'pool_kernel_size': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_out, C_in, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, weight, bias)


def workload(x, weight, bias):
    k_pool = CONFIG['pool_kernel_size']
    sub_val = CONFIG['subtract_value']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    out = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + bias.reshape(1, 1, 1, -1)
    out = out - sub_val
    out = jax.nn.hard_swish(out)
    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, k_pool, k_pool, 1),
        window_strides=(1, k_pool, k_pool, 1),
        padding='VALID',
    )
    out = out * jnp.tanh(jax.nn.softplus(out))
    out_nchw = jnp.transpose(out, (0, 3, 1, 2))
    return out_nchw
