"""DecInit: flatten -> Linear -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DecInit_14710',
    'batch': 8192,
    'd_enc': 4096,
    'n_enc_layer': 1,
    'd_dec': 4096,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    d_in = CONFIG['d_enc'] * CONFIG['n_enc_layer']
    x = jax.random.uniform(kx, (CONFIG['batch'], d_in), dtype)
    w = jax.random.normal(kw, (d_in, CONFIG['d_dec']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['d_dec'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    h = x.reshape(x.shape[0], -1)
    return jnp.tanh(h @ w + b)
