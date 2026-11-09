"""[m1 M2_second_output variant of kb/l2/82_Conv2d_Tanh_Scaling_BiasAdd_Max] Conv2d -> tanh -> scaling -> bias add -> MaxPool2d, also returning the per-example absmax of the pooled output (a per-sample quantization scale)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '82_Conv2d_Tanh_Scaling_BiasAdd_Max_m1_aux',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'scaling_factor': 2.0,
    'pool_kernel_size': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb, kbias = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    Cin = CONFIG['in_channels']
    Cout = CONFIG['out_channels']
    H, W = CONFIG['height'], CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(kx, (B, Cin, H, W), dtype)
    conv_w = jax.random.normal(kw, (Cout, Cin, K, K), dtype) * 0.02
    conv_b = jax.random.normal(kb, (Cout,), dtype) * 0.02
    bias = jax.random.normal(kbias, (Cout, 1, 1), dtype) * 0.02
    return x, conv_w, conv_b, bias


def workload(x, conv_w, conv_b, bias):
    p = CONFIG['pool_kernel_size']
    s = CONFIG['scaling_factor']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + conv_b.reshape(1, 1, 1, -1)
    y = jnp.tanh(y)
    y = y * s
    y = y + bias.reshape(1, 1, 1, -1)
    y = lax.reduce_window(
        y, -jnp.inf, lax.max,
        window_dimensions=(1, p, p, 1),
        window_strides=(1, p, p, 1),
        padding='VALID',
    )
    absmax = jnp.max(jnp.abs(y), axis=(1, 2, 3))
    return jnp.transpose(y, (0, 3, 1, 2)), absmax
