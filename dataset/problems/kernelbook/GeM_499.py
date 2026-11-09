"""[bf16 variant] GeM pooling: clamp(min=eps) -> pow(p) -> global avg pool -> pow(1/p)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeM_499',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'p': 3.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    p = jnp.ones((1,), dtype) * CONFIG['p']
    return x, p


def workload(x, p):
    # avg_pool2d(x, H, W) is a full-plane mean only when H == W (kernel HxH, stride W).
    assert x.shape[-2] == x.shape[-1], 'GeM port assumes square spatial dims'
    p3 = p.reshape(1, 1, 1)
    x = jnp.maximum(x, CONFIG['eps'])
    x = jnp.power(x, p3)
    x = jnp.mean(x, axis=(-2, -1), keepdims=True)
    return jnp.power(x, 1.0 / p3)
