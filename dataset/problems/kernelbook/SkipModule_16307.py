"""SkipModule: linear1 -> linear2 -> concat(x, x1) -> linear3 (all with bias, no activation)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SkipModule_16307',
    'batch': 16,
    'dim1': 64,
    'dim2': 64,
    'in_features': 1024,
    'out_features': 1024,
}


def create_inputs(dtype=jnp.float32):
    b, d1, d2 = CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2']
    fin, fout = CONFIG['in_features'], CONFIG['out_features']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (b, d1, d2, fin), dtype)
    w1 = jax.random.normal(keys[1], (fin, fout), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (fout,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (fout, fout), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (fout,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (fin + fout, fout), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (fout,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x1 = x @ w1 + b1
    x1 = x1 @ w2 + b2
    cat = jnp.concatenate((x, x1), axis=-1)
    return cat @ w3 + b3
