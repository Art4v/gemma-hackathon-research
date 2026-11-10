"""FiLMLayer_PreSin: out = (style @ Wg + bg) * sin(x @ W + b) + (style @ Wb + bb)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FiLMLayer_PreSin_6999',
    'batch': 8192,
    'in_dim': 2048,
    'out_dim': 2048,
    'style_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    b, i, o, s = CONFIG['batch'], CONFIG['in_dim'], CONFIG['out_dim'], CONFIG['style_dim']
    k = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(k[0], (b, i), dtype)
    style = jax.random.uniform(k[1], (b, s), dtype)
    w = jax.random.normal(k[2], (i, o), dtype) * 0.02
    bl = jax.random.normal(k[3], (o,), dtype) * 0.02
    wg = jax.random.normal(k[4], (s, o), dtype) * 0.02
    bg = jnp.ones((o,), dtype)
    wb = jax.random.normal(k[5], (s, o), dtype) * 0.02
    bb = jnp.zeros((o,), dtype)
    return x, style, w, bl, wg, bg, wb, bb


def workload(x, style, w, bl, wg, bg, wb, bb):
    gain = style @ wg + bg
    bias = style @ wb + bb
    return gain * jnp.sin(x @ w + bl) + bias
