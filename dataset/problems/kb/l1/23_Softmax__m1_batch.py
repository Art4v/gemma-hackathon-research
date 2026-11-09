"""[m1 M1_batch variant of kb/l1/23_Softmax] Numerically stable softmax over the last axis, applied to a leading batch of 4 independent (256, 32768) inputs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '23_Softmax_m1_batch',
    'num_batches': 4,
    'batch_size': 256,
    'dim': 32768,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['num_batches'], CONFIG['batch_size'], CONFIG['dim']), dtype=dtype
    )
    return (x,)


def workload(x):
    # torch.softmax(x, dim=-1): numerically stable softmax along the last axis.
    x_max = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - x_max)
    return e / jnp.sum(e, axis=-1, keepdims=True)
