"""LinearBlock_67: Linear (x @ w + b) on the last axis followed by ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearBlock_67',
    'batch': 8,
    'height': 128,
    'width': 128,
    'input_dim': 512,
    'output_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['input_dim']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['input_dim'], CONFIG['output_dim']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['output_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    return jax.nn.relu(x @ w + b)
