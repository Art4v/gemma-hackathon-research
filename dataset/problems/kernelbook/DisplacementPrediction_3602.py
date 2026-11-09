"""DisplacementPrediction: per-pedestrian shared Linear (x @ W + b), stacked on axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DisplacementPrediction_3602',
    'batch': 16,
    'pedestrian_num': 128,
    'seq': 128,
    'input_size': 256,
    'output_size': 256,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    data = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['pedestrian_num'], CONFIG['seq'], CONFIG['input_size']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['input_size'], CONFIG['output_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['output_size'],), dtype) * 0.02
    return data, w, b


def workload(data, w, b):
    # The per-pedestrian loop plus stack on axis 1 is one linear layer over the last axis.
    return data @ w + b
