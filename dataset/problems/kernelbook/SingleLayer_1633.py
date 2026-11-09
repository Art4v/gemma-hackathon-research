"""GroupNorm(C groups) -> ReLU -> Conv2d 3x3 (no bias) -> concat with input along channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SingleLayer_1633',
    'batch': 8,
    'n_channels': 128,
    'growth_rate': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (c['batch'], c['n_channels'], c['height'], c['width']), dtype)
    conv_w = jax.random.normal(
        kw, (c['growth_rate'], c['n_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    gamma = jnp.ones((c['n_channels'],), dtype)
    beta = jnp.zeros((c['n_channels'],), dtype)
    return x, conv_w, gamma, beta


def workload(x, conv_w, gamma, beta):
    # GroupNorm with num_groups == num_channels: normalize each (n, c) over H, W.
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(2, 3), keepdims=True)
    h = (x - mean) * lax.rsqrt(var + CONFIG['eps'])
    h = h * gamma[None, :, None, None] + beta[None, :, None, None]
    h = jnp.maximum(h, 0)

    kh, kw = conv_w.shape[2], conv_w.shape[3]
    out = lax.conv_general_dilated(
        jnp.transpose(h, (0, 2, 3, 1)),
        jnp.transpose(conv_w, (2, 3, 1, 0)),
        window_strides=(1, 1),
        padding=((kh // 2, kh // 2), (kw // 2, kw // 2)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = jnp.transpose(out, (0, 3, 1, 2))
    return jnp.concatenate([x, out], axis=1)
