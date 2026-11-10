"""[bf16 variant] Depthwise Conv2d (groups=nin, k=4, pad=4) -> pointwise 1x1 Conv2d."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'depthwise_separable_conv_7238',
    'batch': 16,
    'nin': 128,
    'nout': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'padding': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k1, (c['batch'], c['nin'], c['height'], c['width']), dtype)
    dw_w = jax.random.normal(k2, (c['kernel_size'], c['kernel_size'], 1, c['nin']), dtype) * 0.02
    dw_b = jax.random.normal(k3, (c['nin'],), dtype) * 0.02
    pw_w = jax.random.normal(k4, (c['nin'], c['nout']), dtype) * 0.02
    pw_b = jax.random.normal(k5, (c['nout'],), dtype) * 0.02
    return x, dw_w, dw_b, pw_w, pw_b


def workload(x, dw_w, dw_b, pw_w, pw_b):
    p = CONFIG['padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = lax.conv_general_dilated(
        h, dw_w,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=x.shape[1],
    ) + dw_b
    h = h @ pw_w + pw_b
    return jnp.transpose(h, (0, 3, 1, 2))
