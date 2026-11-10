"""[bf16 variant] Feedforward: Linear -> ReLU -> Linear -> Sigmoid over the last axis of a rank-4 tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Feedforward_9337',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 512,
    'hidden_size': 512,
    'output_size': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_size']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (CONFIG['input_size'], CONFIG['hidden_size']), dtype) * scale
    b1 = jax.random.normal(k_b1, (CONFIG['hidden_size'],), dtype) * scale
    w2 = jax.random.normal(k_w2, (CONFIG['hidden_size'], CONFIG['output_size']), dtype) * scale
    b2 = jax.random.normal(k_b2, (CONFIG['output_size'],), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    hidden = x @ w1 + b1
    activated = jnp.maximum(hidden, 0)
    output = activated @ w2 + b2
    return jax.nn.sigmoid(output)
