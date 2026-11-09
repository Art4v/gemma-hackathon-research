"""[bf16 variant] Linear Q/K/V -> raw view to (B,H,T,hd) -> softmax(Q^T K / sqrt(hd)) -> (.) V^T -> raw view -> Linear projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_7562',
    'batch': 8,
    'seq_len': 2048,
    'dim': 1024,
    'heads': 32,
    'hidden_dim': 32,
}


def create_inputs(dtype=jnp.bfloat16):
    b, t, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    hh = CONFIG['heads'] * CONFIG['hidden_dim']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (b, t, d), dtype)
    wq = jax.random.normal(keys[1], (d, hh), dtype) * 0.02
    bq = jax.random.normal(keys[2], (hh,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (d, hh), dtype) * 0.02
    bk = jax.random.normal(keys[4], (hh,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (d, hh), dtype) * 0.02
    bv = jax.random.normal(keys[6], (hh,), dtype) * 0.02
    wp = jax.random.normal(keys[7], (hh, d), dtype) * 0.02
    bp = jax.random.normal(keys[8], (d,), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv, wp, bp


def workload(x, wq, bq, wk, bk, wv, bv, wp, bp):
    b, t, _ = x.shape
    hh = wq.shape[1]
    # CONFIG keeps heads == hidden_dim, so hd is the square root of the
    # projection width. This also fits the stock torch Model (heads=hidden_dim=4).
    hd = int(round(hh ** 0.5))
    h = hh // hd
    # Raw reshape like torch .view: (B, T, H*hd) -> (B, H, T, hd), no transpose.
    q = (x @ wq + bq).reshape(b, h, t, hd)
    k = (x @ wk + bk).reshape(b, h, t, hd)
    v = (x @ wv + bv).reshape(b, h, t, hd)
    scores = jnp.einsum('bhtd,bhte->bhde', q, k) / (hd ** 0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('bhde,bhte->bhdt', weights, v)
    # Raw reshape of the contiguous (B, H, hd, T) buffer, like torch .view.
    out = out.reshape(b, t, h * hd)
    return out @ wp + bp
