"""AFT-Simple: Linear Q/K/V -> sigmoid(Q) * softmax(K, -1) * V -> Linear projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AFTSimple_2134',
    'batch': 16,
    'seq_len': 2048,
    'dim': 1024,
    'hidden_dim': 128,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    b, t = CONFIG['batch'], CONFIG['seq_len']
    d, h = CONFIG['dim'], CONFIG['hidden_dim']
    x = jax.random.uniform(keys[0], (b, t, d), dtype)
    wq = jax.random.normal(keys[1], (d, h), dtype) * 0.02
    bq = jax.random.normal(keys[2], (h,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (d, h), dtype) * 0.02
    bk = jax.random.normal(keys[4], (h,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (d, h), dtype) * 0.02
    bv = jax.random.normal(keys[6], (h,), dtype) * 0.02
    wp = jax.random.normal(keys[7], (h, d), dtype) * 0.02
    bp = jax.random.normal(keys[8], (d,), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv, wp, bp


def workload(x, wq, bq, wk, bk, wv, bv, wp, bp):
    q = x @ wq + bq
    k = x @ wk + bk
    v = x @ wv + bv
    yt = jax.nn.sigmoid(q) * (jax.nn.softmax(k, axis=-1) * v)
    return yt @ wp + bp
