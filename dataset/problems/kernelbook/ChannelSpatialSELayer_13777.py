"""ChannelSpatialSELayer: max(channel SE (mean -> fc -> relu -> fc -> sigmoid -> scale), spatial SE (1x1 conv -> sigmoid -> scale))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ChannelSpatialSELayer_13777',
    'batch': 16,
    'num_channels': 256,
    'reduction_ratio': 2,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    b, c = CONFIG['batch'], CONFIG['num_channels']
    cr = c // CONFIG['reduction_ratio']
    h, w = CONFIG['height'], CONFIG['width']
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (b, c, h, w), dtype)
    w1 = jax.random.normal(ks[1], (c, cr), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (cr,), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (cr, c), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c,), dtype) * 0.02
    wc = jax.random.normal(ks[5], (1, 1, c, 1), dtype) * 0.02
    bc = jax.random.normal(ks[6], (1,), dtype) * 0.02
    return x, w1, b1, w2, b2, wc, bc


def workload(x, w1, b1, w2, b2, wc, bc):
    # Channel SE
    s = jnp.mean(x, axis=(2, 3))
    hdn = jax.nn.relu(s @ w1 + b1)
    g = jax.nn.sigmoid(hdn @ w2 + b2)
    cse = x * g[:, :, None, None]
    # Spatial SE: 1x1 conv C -> 1
    xh = jnp.transpose(x, (0, 2, 3, 1))
    out = lax.conv_general_dilated(
        xh, wc, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    sp = jax.nn.sigmoid(out + bc)
    sse = x * jnp.transpose(sp, (0, 3, 1, 2))
    return jnp.maximum(cse, sse)
