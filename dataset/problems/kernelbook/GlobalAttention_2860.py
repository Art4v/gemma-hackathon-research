"""GlobalAttention: add -> global avg pool -> 1x1 conv (+bias) -> sigmoid -> gated blend of x and y."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GlobalAttention_2860',
    'batch': 8,
    'channels': 256,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    kx, ky, kw, kb = jax.random.split(jax.random.key(0), 4)
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    y = jax.random.uniform(ky, (b, c, h, w), dtype)
    # 1x1 conv weight stored as (C_in, C_out); bias (C_out,)
    weight = jax.random.normal(kw, (c, c), dtype) * 0.02
    bias = jax.random.normal(kb, (c,), dtype) * 0.02
    return x, y, weight, bias


def workload(x, y, weight, bias):
    # ponytail: 1x1 conv on a 1x1 pooled map is exactly a dense layer, so no conv/NHWC round-trip.
    pooled = jnp.mean(x + y, axis=(2, 3))            # (B, C)
    att = jax.nn.sigmoid(pooled @ weight + bias)      # (B, C)
    att = att[:, :, None, None]
    return x * att + y * (1 - att)
