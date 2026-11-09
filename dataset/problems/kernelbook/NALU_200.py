"""NALU: sigmoid gate, NAC linear (tanh(W)*sigmoid(M)) add path, and exp(NAC(log|x|+eps)) multiply path."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NALU_200',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'n_in': 1024,
    'n_out': 1024,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_m, k_g = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_in']), dtype)
    w_hat = jax.random.normal(k_w, (CONFIG['n_in'], CONFIG['n_out']), dtype) * 0.02
    m_hat = jax.random.normal(k_m, (CONFIG['n_in'], CONFIG['n_out']), dtype) * 0.02
    g = jax.random.normal(k_g, (CONFIG['n_in'], 1), dtype) * 0.02
    return x, w_hat, m_hat, g


def workload(x, w_hat, m_hat, g_w):
    w = jnp.tanh(w_hat) * jax.nn.sigmoid(m_hat)
    g = jax.nn.sigmoid(x @ g_w)
    add_path = x @ w
    mul_path = jnp.exp(jnp.log(jnp.abs(x) + CONFIG['eps']) @ w)
    return g * add_path + (1 - g) * mul_path
