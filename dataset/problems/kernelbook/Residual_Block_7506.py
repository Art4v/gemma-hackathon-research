"""[bf16 variant] Residual_Block_7506: CoordConv(with_r) -> InstanceNorm -> LeakyReLU -> CoordConv(with_r) -> InstanceNorm -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Residual_Block_7506',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'coord_channels': 3,  # xx, yy, rr (with_r=True)
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'eps': 1e-5,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG['channels']
    cin = c + CONFIG['coord_channels']
    k = CONFIG['kernel_size']
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, cin, c), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, cin, c), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c,), dtype) * 0.02
    g1 = jnp.ones((c,), dtype)
    be1 = jnp.zeros((c,), dtype)
    g2 = jnp.ones((c,), dtype)
    be2 = jnp.zeros((c,), dtype)
    return x, w1, b1, g1, be1, w2, b2, g2, be2


def _add_coords(x):
    # x is NHWC. PyTorch's AddCoords makes xx vary along H and yy along W.
    b, h, w, _ = x.shape
    xx = jnp.arange(h, dtype=jnp.float32) / (h - 1) * 2 - 1
    yy = jnp.arange(w, dtype=jnp.float32) / (w - 1) * 2 - 1
    xx = jnp.broadcast_to(xx[None, :, None, None], (b, h, w, 1)).astype(x.dtype)
    yy = jnp.broadcast_to(yy[None, None, :, None], (b, h, w, 1)).astype(x.dtype)
    rr = jnp.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    return jnp.concatenate([x, xx, yy, rr], axis=-1)


def _coord_conv(x, w, b):
    s = CONFIG['stride']
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        _add_coords(x), w, window_strides=(s, s), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _instance_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, b1, g1, be1, w2, b2, g2, be2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = _instance_norm(_coord_conv(h, w1, b1), g1, be1)
    out = jnp.where(out >= 0, out, out * CONFIG['negative_slope'])
    out = _instance_norm(_coord_conv(out, w2, b2), g2, be2)
    out = out + h
    return jnp.transpose(out, (0, 3, 1, 2))
