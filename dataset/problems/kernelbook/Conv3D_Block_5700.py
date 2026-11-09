"""Conv3d (1x3x3 kernel, padding 1 on all dims) + bias -> ReLU -> MaxPool3d (1,2,2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv3D_Block_5700',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'time': 4,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'time_padding': 1,
    'pool_size': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['time'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w, (c['out_channels'], c['in_channels'], 1, c['kernel_size'], c['kernel_size']),
        dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    # KernelBench feeds an unbatched (C,T,H,W) tensor; Conv3d accepts that, so mirror it.
    unbatched = x.ndim == 4
    if unbatched:
        x = x[None]
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(w, (2, 3, 4, 1, 0))
    s, p, tp = c['stride'], c['padding'], c['time_padding']
    y = lax.conv_general_dilated(
        x_ndhwc, w_dhwio,
        window_strides=(1, s, s),
        padding=((tp, tp), (p, p), (p, p)),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = jnp.maximum(y + b, 0)
    ps = c['pool_size']
    y = lax.reduce_window(
        y, -jnp.inf, lax.max,
        window_dimensions=(1, 1, ps, ps, 1),
        window_strides=(1, 1, ps, ps, 1),
        padding='VALID',
    )
    y = jnp.transpose(y, (0, 4, 1, 2, 3))
    return y[0] if unbatched else y
