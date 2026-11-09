"""FeatureDiscriminator: 5x (1x1 conv) with LeakyReLU x4 + Sigmoid, then AvgPool(1,25) and flatten."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FeatureDiscriminator_17427',
    'batch': 32,
    'in_channels': 512,
    'height': 64,
    'width': 64,
    'channels': (512, 256, 128, 64, 32, 1),
    'negative_slope': 0.01,
    'pool_kernel': (1, 25),
}


def create_inputs(dtype=jnp.float32):
    chans = CONFIG['channels']
    n_layers = len(chans) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i in range(n_layers):
        cin, cout = chans[i], chans[i + 1]
        w = jax.random.normal(keys[1 + 2 * i], (1, 1, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def _conv1x1(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, *params):
    slope = CONFIG['negative_slope']
    kh, kw = CONFIG['pool_kernel']
    h = jnp.transpose(x, (0, 2, 3, 1))
    ws, bs = params[0::2], params[1::2]
    for w, b in zip(ws[:-1], bs[:-1]):
        h = jax.nn.leaky_relu(_conv1x1(h, w, b), negative_slope=slope)
    h = jax.nn.sigmoid(_conv1x1(h, ws[-1], bs[-1]))
    h = lax.reduce_window(
        h, jnp.array(0, h.dtype), lax.add,
        window_dimensions=(1, kh, kw, 1), window_strides=(1, 1, 1, 1), padding='VALID',
    ) / (kh * kw)
    return jnp.transpose(h, (0, 3, 1, 2)).reshape(-1)
