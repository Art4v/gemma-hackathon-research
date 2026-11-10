"""TemporalDecayRegression: matmul (x @ w) -> relu -> negate -> exp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TemporalDecayRegression_3216',
    'batch': 16,
    'height': 128,
    'width': 128,
    'input_size': 128,
    'output_size': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['input_size']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['input_size'], CONFIG['output_size']),
        dtype,
    ) * CONFIG['weight_scale']
    return x, w


def workload(x, w):
    return jnp.exp(-jax.nn.relu(x @ w))
