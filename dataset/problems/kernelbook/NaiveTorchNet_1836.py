"""[bf16 variant] NaiveTorchNet: Linear (no bias) -> sigmoid -> Linear (no bias) -> sigmoid over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NaiveTorchNet_1836',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_nodes': 256,
    'hidden_nodes': 256,
    'output_nodes': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_hidden, k_output = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_nodes']),
        dtype,
    )
    w_hidden = jax.random.normal(
        k_hidden, (CONFIG['input_nodes'], CONFIG['hidden_nodes']), dtype
    ) * CONFIG['weight_scale']
    w_output = jax.random.normal(
        k_output, (CONFIG['hidden_nodes'], CONFIG['output_nodes']), dtype
    ) * CONFIG['weight_scale']
    return x, w_hidden, w_output


def workload(x, w_hidden, w_output):
    h = jax.nn.sigmoid(x @ w_hidden)
    return jax.nn.sigmoid(h @ w_output)
