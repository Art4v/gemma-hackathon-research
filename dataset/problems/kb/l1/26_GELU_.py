"""Exact (erf) GELU activation over a 2-D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '26_GELU_',
    'batch_size': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype)
    return (x,)


def workload(x):
    # torch.nn.functional.gelu default is the exact erf formulation.
    return jax.nn.gelu(x, approximate=False)
