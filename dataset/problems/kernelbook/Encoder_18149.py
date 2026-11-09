"""[bf16 variant] Conv2d(4x4, stride 1, pad 1) -> InstanceNorm2d -> LeakyReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Encoder_18149',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': (4, 4),
    'stride': 1,
    'padding': (1, 1),  # (k - 1) // 2 per spatial dim, as in the torch module
    'eps': 1e-5,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    kh, kw = CONFIG['kernel_size']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (kh, kw, CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    ph, pw = CONFIG['padding']
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=(s, s),
        padding=((ph, ph), (pw, pw)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    # InstanceNorm2d: per-sample, per-channel stats over spatial dims, biased variance, no affine.
    mean = jnp.mean(y, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(1, 2), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + CONFIG['eps'])
    y = jnp.where(y >= 0, y, y * CONFIG['negative_slope'])
    return jnp.transpose(y, (0, 3, 1, 2))
