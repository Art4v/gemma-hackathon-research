"""Conv3d(k=3, pad=1) -> InstanceNorm3d (no affine) -> ELU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LUConv_11290',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'padding': 1,
    'eps': 1e-5,
    'elu_alpha': 1.0,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (k, k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # ponytail: accept PyTorch's unbatched (C,D,H,W) form too, so the output shape matches torch.
    unbatched = x.ndim == 4
    if unbatched:
        x = x[None]
    p = CONFIG['padding']
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    y = lax.conv_general_dilated(
        xh, w,
        window_strides=(1, 1, 1),
        padding=[(p, p)] * 3,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    mean = jnp.mean(y, axis=(1, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(1, 2, 3), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + CONFIG['eps'])
    y = jax.nn.elu(y, alpha=CONFIG['elu_alpha'])
    out = jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
    return out[0] if unbatched else out
