"""[bf16 variant] Softmax over the feature (last) axis of a 2-D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '23_Softmax',
    'batch_size': 1024,
    'dim': 32768,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype
    )
    return (x,)


def workload(x):
    # torch.softmax(x, dim=1): numerically stable softmax along the last axis.
    x_max = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - x_max)
    return e / jnp.sum(e, axis=-1, keepdims=True)
