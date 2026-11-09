"""GLU_1348: channel-wise Linear (NCHW -> NHWC matmul -> NCHW) times sigmoid(x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GLU_1348',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG['channels']
    x = jax.random.uniform(kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (c, c), dtype) * 0.02
    b = jax.random.normal(kb, (c,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    lin = jnp.transpose(x_nhwc @ w + b, (0, 3, 1, 2))
    return lin * jax.nn.sigmoid(x)
