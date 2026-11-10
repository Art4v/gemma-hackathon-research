"""[bf16 variant] Net16_3522: Linear -> ReLU -> Linear applied over the last axis of a rank-4 tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net16_3522',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_dim': 256,
    'hidden_dim': 128,
    'output_dim': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_dim']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (CONFIG['input_dim'], CONFIG['hidden_dim']), dtype) * CONFIG['weight_scale']
    b1 = jax.random.normal(k_b1, (CONFIG['hidden_dim'],), dtype) * CONFIG['weight_scale']
    w2 = jax.random.normal(k_w2, (CONFIG['hidden_dim'], CONFIG['output_dim']), dtype) * CONFIG['weight_scale']
    b2 = jax.random.normal(k_b2, (CONFIG['output_dim'],), dtype) * CONFIG['weight_scale']
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
