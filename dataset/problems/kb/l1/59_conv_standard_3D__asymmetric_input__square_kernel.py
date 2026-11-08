"""Conv3d with (k, k, 1) kernel, no bias, stride 1, padding 0 (NCDHW interface)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '59_conv_standard_3D__asymmetric_input__square_kernel',
    'batch_size': 16,
    'in_channels': 3,
    'out_channels': 64,
    'kernel_size': 3,
    'kernel_depth': 1,
    'height': 256,
    'width': 256,
    'depth': 10,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'],
         CONFIG['width'], CONFIG['depth']),
        dtype=dtype,
    )
    w = jax.random.normal(
        kw,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'],
         CONFIG['kernel_size'], CONFIG['kernel_depth']),
        dtype=dtype,
    ) * 0.02
    return (x, w)


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    x_cl = jnp.transpose(x, (0, 2, 3, 4, 1))          # N,H,W,D,C
    w_cl = jnp.transpose(w, (2, 3, 4, 1, 0))          # kH,kW,kD,I,O
    y = lax.conv_general_dilated(
        x_cl, w_cl,
        window_strides=(s, s, s),
        padding=((p, p), (p, p), (p, p)),
        rhs_dilation=(d, d, d),
        dimension_numbers=('NHWDC', 'HWDIO', 'NHWDC'),
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))           # N,O,H,W,D
