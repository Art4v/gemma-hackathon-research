"""[bf16 variant] Residual block: conv3x3 -> ReLU -> conv3x3 -> scale(0.1) -> add identity."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '_Residual_Block_16661',
    'batch': 8,
    'channels': 256,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'res_scale': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    b, c = CONFIG['batch'], CONFIG['channels']
    h, w, ks = CONFIG['height'], CONFIG['width'], CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    w1 = jax.random.normal(k_w1, (ks, ks, c, c), dtype) * 0.02  # HWIO
    w2 = jax.random.normal(k_w2, (ks, ks, c, c), dtype) * 0.02  # HWIO
    return x, w1, w2


def _conv(x_nhwc, w_hwio):
    s, p = CONFIG['stride'], CONFIG['padding']
    return lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )


def workload(x, w1, w2):
    # Residual add requires the block to preserve channel count.
    assert w2.shape[3] == x.shape[1], 'conv2 out channels must equal input channels'
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_conv(x_nhwc, w1))
    out = _conv(out, w2) * CONFIG['res_scale']
    return jnp.transpose(out, (0, 3, 1, 2)) + x
