"""SE_layer_3d: global avg pool 3d -> linear -> relu -> linear -> sigmoid -> channel-wise scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SE_layer_3d_4495',
    'batch': 8,
    'num_channels': 128,
    'reduction_ratio': 2,
    'depth': 32,
    'height': 32,
    'width': 32,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    c = CONFIG['num_channels']
    cr = c // CONFIG['reduction_ratio']
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (b, c, CONFIG['depth'], CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k[1], (c, cr), dtype) * 0.02
    b1 = jax.random.normal(k[2], (cr,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (cr, c), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    squeeze = jnp.mean(x, axis=(2, 3, 4)).astype(jnp.float32)
    h = jax.nn.relu(squeeze @ w1 + b1)
    s = jax.nn.sigmoid(h @ w2 + b2)
    return x * s[:, :, None, None, None]
