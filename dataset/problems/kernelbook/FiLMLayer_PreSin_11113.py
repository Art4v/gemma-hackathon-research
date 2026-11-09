"""FiLM layer: Linear -> sin, then gain/bias from style Linears: gain * sin(x W + b) + bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FiLMLayer_PreSin_11113',
    'batch': 8192,
    'in_dim': 2048,
    'out_dim': 2048,
    'style_dim': 2048,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    i, o, s = CONFIG['in_dim'], CONFIG['out_dim'], CONFIG['style_dim']
    kx, ks, kw, kg, kb = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (b, i), dtype)
    style = jax.random.uniform(ks, (b, s), dtype)
    w_lin = jax.random.normal(kw, (i, o), dtype) * 0.02
    b_lin = jnp.zeros((o,), dtype)
    w_gain = jax.random.normal(kg, (s, o), dtype) * 0.02
    b_gain = jnp.ones((o,), dtype)
    w_bias = jax.random.normal(kb, (s, o), dtype) * 0.02
    b_bias = jnp.zeros((o,), dtype)
    return x, style, w_lin, b_lin, w_gain, b_gain, w_bias, b_bias


def workload(x, style, w_lin, b_lin, w_gain, b_gain, w_bias, b_bias):
    gain = style @ w_gain + b_gain
    bias = style @ w_bias + b_bias
    h = jnp.sin(x @ w_lin + b_lin)
    return gain * h + bias
