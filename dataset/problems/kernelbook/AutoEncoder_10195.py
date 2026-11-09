"""[bf16 variant] AutoEncoder: 4x(conv3x3+relu+maxpool2 ceil) -> conv+relu -> nearest up(75) -> 3x(conv+relu+nearest up x2) -> conv3x3."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AutoEncoder_10195',
    'batch': 32,
    'in_channels': 3,
    'height': 512,
    'width': 512,
    'encoder_channels': (3, 16, 8, 4, 4),
    'decoder_channels': (4, 4, 4, 8, 16, 3),
    'kernel_size': 3,
    'padding': 1,
    'pool_size': 2,
    'up1_size': 75,
    'up_scale': 2,
}


def _layer_shapes():
    k = CONFIG['kernel_size']
    shapes = []
    for chans in (CONFIG['encoder_channels'], CONFIG['decoder_channels']):
        for cin, cout in zip(chans[:-1], chans[1:]):
            shapes.append(((cout, cin, k, k), (cout,)))
    return shapes


def create_inputs(dtype=jnp.bfloat16):
    shapes = _layer_shapes()
    key = jax.random.key(0)
    keys = jax.random.split(key, 1 + 2 * len(shapes))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i, (w_shape, b_shape) in enumerate(shapes):
        params.append(jax.random.normal(keys[1 + 2 * i], w_shape, dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], b_shape, dtype) * 0.02)
    return (x, *params)


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _pool_ceil(x):
    # ceil_mode with kernel == stride: pad -inf at the end up to a multiple of k.
    k = CONFIG['pool_size']
    ph = (-x.shape[1]) % k
    pw = (-x.shape[2]) % k
    init = jnp.array(-jnp.inf, x.dtype)
    return lax.reduce_window(
        x, init, lax.max, (1, k, k, 1), (1, k, k, 1),
        ((0, 0), (0, ph), (0, pw), (0, 0)),
    )


def _nearest_idx(in_size, out_size):
    # PyTorch nearest: min(floor(dst * float32(in/out)), in - 1).
    scale = jnp.float32(in_size) / jnp.float32(out_size)
    idx = jnp.floor(jnp.arange(out_size, dtype=jnp.float32) * scale).astype(jnp.int32)
    return jnp.minimum(idx, in_size - 1)


def _resize_nearest(x, size):
    x = jnp.take(x, _nearest_idx(x.shape[1], size), axis=1)
    return jnp.take(x, _nearest_idx(x.shape[2], size), axis=2)


def _upscale(x):
    s = CONFIG['up_scale']
    return jnp.repeat(jnp.repeat(x, s, axis=1), s, axis=2)


def workload(x, *params):
    ws, bs = params[0::2], params[1::2]
    n_enc = len(CONFIG['encoder_channels']) - 1
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i in range(n_enc):
        h = _pool_ceil(jax.nn.relu(_conv(h, ws[i], bs[i])))
    dw, db = ws[n_enc:], bs[n_enc:]
    h = jax.nn.relu(_conv(h, dw[0], db[0]))
    h = _resize_nearest(h, CONFIG['up1_size'])
    for i in range(1, len(dw) - 1):
        h = _upscale(jax.nn.relu(_conv(h, dw[i], db[i])))
    h = _conv(h, dw[-1], db[-1])
    return jnp.transpose(h, (0, 3, 1, 2))
