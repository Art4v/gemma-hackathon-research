"""[bf16 variant] TimeEncode: reshape(-1, 1) -> Linear(1, dim) -> cos."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TimeEncode_18298',
    't_shape': (8, 128, 128, 96),
    'dim': 128,
    'freq_exp_max': 9.0,
}


def create_inputs(dtype=jnp.bfloat16):
    t = jax.random.uniform(jax.random.key(0), CONFIG['t_shape'], dtype)
    dim = CONFIG['dim']
    # Module's fixed init: weight = 1 / 10**linspace(0, 9, dim), stored (in=1, out=dim); bias = 0.
    w = (1.0 / 10.0 ** jnp.linspace(0.0, CONFIG['freq_exp_max'], dim, dtype=jnp.float32)).astype(dtype).reshape(1, dim)
    b = jnp.zeros((dim,), dtype)
    return t, w, b


def workload(t, w, b):
    # ponytail: in_features == 1, so x @ w is an outer product; broadcast keeps it exact f32.
    x = t.reshape(-1, 1)
    return jnp.cos(x * w + b)
