"""ConvTranspose2d(k2,s2)+ReLU x5 decoder (256->128->64->32->16->3)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvDecoder_13343',
    'batch': 16,
    'height': 64,
    'width': 64,
    'channels': [256, 128, 64, 32, 16, 3],
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    ch = CONFIG['channels']
    k = CONFIG['kernel_size']
    n_layers = len(ch) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], ch[0], CONFIG['height'], CONFIG['width']), dtype)
    params = []
    for i in range(n_layers):
        # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
        w = jax.random.normal(keys[1 + 2 * i], (ch[i], ch[i + 1], k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (ch[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def _conv_transpose(x, w, b):
    # x: NHWC, w: (in, out, kH, kW) PyTorch layout.
    s = CONFIG['stride']
    p = CONFIG['padding']
    kh, kw = w.shape[2], w.shape[3]
    k_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x, k_hwio,
        window_strides=(1, 1),
        padding=((kh - 1 - p, kh - 1 - p), (kw - 1 - p, kw - 1 - p)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, *params):
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i in range(0, len(params), 2):
        h = jax.nn.relu(_conv_transpose(h, params[i], params[i + 1]))
    return jnp.transpose(h, (0, 3, 1, 2))
