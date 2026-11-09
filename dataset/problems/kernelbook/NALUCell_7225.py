"""[bf16 variant] NALU cell: tanh(W_hat)*sigmoid(M_hat) gated mix of linear (NAC) and exp(NAC(log|x|+eps)), gate = sigmoid(x @ G)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NALUCell_7225',
    'batch': 16,
    'seq': 2048,
    'in_dim': 1024,
    'out_dim': 1024,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, km, kg = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['seq'], CONFIG['in_dim']), dtype)
    wshape = (CONFIG['in_dim'], CONFIG['out_dim'])
    w_hat = jax.random.normal(kw, wshape, dtype) * 0.02
    m_hat = jax.random.normal(km, wshape, dtype) * 0.02
    g_w = jax.random.normal(kg, wshape, dtype) * 0.02
    return x, w_hat, m_hat, g_w


def workload(x, w_hat, m_hat, g_w):
    w = jnp.tanh(w_hat) * jax.nn.sigmoid(m_hat)
    a = x @ w
    g = jax.nn.sigmoid(x @ g_w)
    m = jnp.exp(jnp.log(jnp.abs(x) + CONFIG['eps']) @ w)
    return g * a + (1 - g) * m
