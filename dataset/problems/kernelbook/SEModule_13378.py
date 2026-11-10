"""SEModule: global avgpool -> 1x1 conv -> ReLU -> 1x1 conv -> hardsigmoid -> scale input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SEModule_13378',
    'batch': 16,
    'channel': 512,
    'reduction': 4,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    n, c, h, w = CONFIG['batch'], CONFIG['channel'], CONFIG['height'], CONFIG['width']
    cr = c // CONFIG['reduction']
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (n, c, h, w), dtype)
    w1 = jax.random.normal(k[1], (c, cr), dtype) * 0.02
    b1 = jax.random.normal(k[2], (cr,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (cr, c), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    # 1x1 convs on a 1x1 pooled map are plain matmuls.
    pooled = jnp.mean(x, axis=(2, 3))
    hdn = jax.nn.relu(pooled @ w1 + b1)
    gate = jnp.clip(hdn @ w2 + b2 + 3.0, 0.0, 6.0) / 6.0
    return x * gate[:, :, None, None]
