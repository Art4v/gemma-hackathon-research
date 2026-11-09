"""[bf16 variant] SEModule: global avg pool -> 1x1 conv -> ReLU -> 1x1 conv -> Hardsigmoid -> channel scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SEModule_8007',
    'batch': 16,
    'in_channels': 512,
    'reduction': 4,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG['in_channels']
    cr = c // CONFIG['reduction']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k1, (c, cr), dtype) * 0.02
    b1 = jax.random.normal(kb1, (cr,), dtype) * 0.02
    w2 = jax.random.normal(k2, (cr, c), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    # 1x1 conv on a 1x1 pooled map is a dense layer, so use matmuls.
    pooled = jnp.mean(x, axis=(2, 3))
    h = jax.nn.relu(pooled @ w1 + b1)
    s = h @ w2 + b2
    s = jnp.clip(s + 3.0, 0.0, 6.0) / 6.0
    return x * s[:, :, None, None]
