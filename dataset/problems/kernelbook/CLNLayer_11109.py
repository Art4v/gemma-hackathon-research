"""[bf16 variant] Linear -> conditional LayerNorm (style-FC gain/bias) -> LeakyReLU(0.2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CLNLayer_11109',
    'batch': 16384,
    'in_dim': 2048,
    'out_dim': 2048,
    'style_dim': 2048,
    'eps': 1e-5,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    b, i, o, s = CONFIG['batch'], CONFIG['in_dim'], CONFIG['out_dim'], CONFIG['style_dim']
    x = jax.random.uniform(keys[0], (b, i), dtype)
    style = jax.random.uniform(keys[1], (b, s), dtype)
    w1 = jax.random.normal(keys[2], (i, o), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (o,), dtype) * 0.02
    wg = jax.random.normal(keys[4], (s, o), dtype) * 0.02
    bg = jax.random.normal(keys[5], (o,), dtype) * 0.02
    wb = jax.random.normal(keys[6], (s, o), dtype) * 0.02
    bb = jax.random.normal(keys[7], (o,), dtype) * 0.02
    return x, style, w1, b1, wg, bg, wb, bb


def workload(x, style, w1, b1, wg, bg, wb, bb):
    h = x @ w1 + b1
    gain = style @ wg + bg + 1.0
    bias = style @ wb + bb
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=-1, keepdims=True)
    out = (h - mean) * jax.lax.rsqrt(var + CONFIG['eps'])
    out = out * gain + bias
    return jnp.where(out >= 0, out, out * CONFIG['negative_slope'])
