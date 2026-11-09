"""Linear -> GELU(exact) -> Linear projection (projection_model)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'projection_model_17910',
    'input_shape': (8, 16, 256, 1024),
    'neo_hidden': 1024,
    'clip_hidden': 512,
}


def create_inputs(dtype=jnp.float32):
    neo = CONFIG['neo_hidden']
    mid = neo // 2
    clip = CONFIG['clip_hidden']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, CONFIG['input_shape'], dtype)
    w1 = jax.random.normal(k1, (neo, mid), dtype) * 0.02
    b1 = jax.random.normal(kb1, (mid,), dtype) * 0.02
    w2 = jax.random.normal(k2, (mid, clip), dtype) * 0.02
    b2 = jax.random.normal(kb2, (clip,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.gelu(x @ w1 + b1, approximate=False)
    return h @ w2 + b2
