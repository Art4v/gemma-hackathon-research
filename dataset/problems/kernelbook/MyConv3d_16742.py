"""[bf16 variant] MyConv3d: replicate pad on depth, then Conv3d with zero padding on H/W, plus bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MyConv3d_16742',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'depth': 8,
    'height': 64,
    'width': 64,
    'kernel_size': 4,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    k = CONFIG['kernel_size']
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['depth'],
         CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        kw, (k, k, k, CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # torch treats a 4D input as unbatched (C, D, H, W) for both the pad and Conv3d.
    unbatched = x.ndim == 4
    if unbatched:
        x = x[None]
    p = (w.shape[0] - 1) // 2
    s = CONFIG['stride']
    x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    x = jnp.pad(x, ((0, 0), (p, p), (0, 0), (0, 0), (0, 0)), mode='edge')
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(s, s, s),
        padding=[(0, 0), (p, p), (p, p)],
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    y = jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
    return y[0] if unbatched else y
