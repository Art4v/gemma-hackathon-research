"""[bf16 variant] Linear -> LeakyReLU -> Linear -> Softmax(axis=0) two-layer network on a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NeuralNetwork_12059',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'in_dim': 512,
    'hidden_dim': 1024,
    'out_dim': 512,
    'negative_slope': 0.01,
    'softmax_axis': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['in_dim']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (CONFIG['in_dim'], CONFIG['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (CONFIG['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (CONFIG['hidden_dim'], CONFIG['out_dim']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (CONFIG['out_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = x @ w1 + b1
    h = jnp.where(h >= 0, h, CONFIG['negative_slope'] * h)
    logits = h @ w2 + b2
    return jax.nn.softmax(logits, axis=CONFIG['softmax_axis'])
