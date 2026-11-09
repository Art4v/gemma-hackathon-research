"""NeuralAccumulatorCell: W = tanh(W_hat) * sigmoid(M_hat); y = x @ W (no bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NeuralAccumulatorCell_7228',
    'batch': 8,
    'heads': 8,
    'seq_len': 512,
    'in_dim': 1024,
    'out_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, km = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['heads'], CONFIG['seq_len'], CONFIG['in_dim']),
        dtype,
    )
    w_hat = jax.random.normal(kw, (CONFIG['in_dim'], CONFIG['out_dim']), dtype) * 0.02
    m_hat = jax.random.normal(km, (CONFIG['in_dim'], CONFIG['out_dim']), dtype) * 0.02
    return x, w_hat, m_hat


def workload(x, w_hat, m_hat):
    # w_hat, m_hat stored (in, out); elementwise transform commutes with the transpose.
    w = jnp.tanh(w_hat) * jax.nn.sigmoid(m_hat)
    return x @ w
