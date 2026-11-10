"""LayerScaling: per-sample flatten -> second moment E[x^2] -> x / sqrt(moment2 + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerScaling_8826',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    n = x.shape[0]
    tmp = jnp.reshape(x, (n, -1))
    moment2 = jnp.mean(tmp * tmp, axis=1, keepdims=True)
    moment2 = jnp.reshape(moment2, (n,) + (1,) * (x.ndim - 1))
    return x / jnp.sqrt(moment2 + CONFIG['eps'])
