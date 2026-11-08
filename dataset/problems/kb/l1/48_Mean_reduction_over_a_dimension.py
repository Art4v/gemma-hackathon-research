"""Mean reduction over a single dimension (torch.mean(x, dim=1))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '48_Mean_reduction_over_a_dimension',
    'batch_size': 16,
    'dim1': 4096,
    'dim2': 1024,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype
    )
    return (x,)


def workload(x):
    return jnp.mean(x, axis=CONFIG['dim'])
