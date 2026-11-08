"""Conv2d -> AvgPool2d -> Sigmoid -> Sum over (C, H, W)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '65_Conv2d_AvgPool_Sigmoid_Sum',
    'batch_size': 32,
    'in_channels': 8,
    'out_channels': 64,
    'height': 384,
    'width': 384,
    'kernel_size': 3,
    'pool_kernel_size': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    Cin = CONFIG['in_channels']
    Cout = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(kx, (B, Cin, H, W), dtype)
    w = jax.random.normal(kw, (K, K, Cin, Cout), dtype) * 0.02  # HWIO
    b = jax.random.normal(kb, (Cout,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['pool_kernel_size']
    x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b.reshape(1, 1, 1, -1)
    y = lax.reduce_window(
        y, jnp.array(0, y.dtype), lax.add,
        window_dimensions=(1, p, p, 1),
        window_strides=(1, p, p, 1),
        padding='VALID',
    ) / (p * p)
    y = jax.nn.sigmoid(y)
    return jnp.sum(y, axis=(1, 2, 3))
