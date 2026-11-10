"""[bf16 variant] Neural Accumulator cell: tanh(W_hat) * sigmoid(M_hat) -> bias-free linear on the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NeuralAccumulatorCell_7926',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'in_dim': 512,
    'out_dim': 512,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_m = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['in_dim']),
        dtype,
    )
    w_shape = (CONFIG['in_dim'], CONFIG['out_dim'])
    w_hat = jax.random.normal(k_w, w_shape, dtype) * CONFIG['weight_scale']
    m_hat = jax.random.normal(k_m, w_shape, dtype) * CONFIG['weight_scale']
    return x, w_hat, m_hat


def workload(x, w_hat, m_hat):
    w = jnp.tanh(w_hat) * jax.nn.sigmoid(m_hat)
    return x @ w
