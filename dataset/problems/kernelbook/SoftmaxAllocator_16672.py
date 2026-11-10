"""[bf16 variant] SoftmaxAllocator: softmax(x / temperature) over the asset dimension (axis 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftmaxAllocator_16672',
    'n_samples': 4096,
    'n_assets': 16384,
    'temperature': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['n_samples'], CONFIG['n_assets']), dtype)
    return (x,)


def workload(x):
    n_samples = x.shape[0]
    temperature = CONFIG['temperature'] * jnp.ones((n_samples,), dtype=x.dtype)
    inp = x / temperature[:, None]
    return jax.nn.softmax(inp, axis=1)
