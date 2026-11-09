"""[bf16 variant] AFT-Full: Linear q/k/v -> exp(wbias) @ (exp(K)*V) / exp(wbias) @ exp(K) -> sigmoid(Q) gate -> Linear project."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'aft_full',
    'batch': 8,
    'seq_len': 2048,
    'max_seqlen': 2048,
    'dim': 1024,
    'hidden_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    b, t, d, h, m = (CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim'],
                     CONFIG['hidden_dim'], CONFIG['max_seqlen'])
    keys = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(keys[0], (b, t, d), dtype)
    wq = jax.random.normal(keys[1], (d, h), dtype) * 0.02
    bq = jax.random.normal(keys[2], (h,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (d, h), dtype) * 0.02
    bk = jax.random.normal(keys[4], (h,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (d, h), dtype) * 0.02
    bv = jax.random.normal(keys[6], (h,), dtype) * 0.02
    wp = jax.random.normal(keys[7], (h, d), dtype) * 0.02
    bp = jax.random.normal(keys[8], (d,), dtype) * 0.02
    wbias = jax.random.normal(keys[9], (m, m), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv, wp, bp, wbias


def workload(x, wq, bq, wk, bk, wv, bv, wp, bp, wbias):
    t = x.shape[1]
    q = x @ wq + bq
    k = x @ wk + bk
    v = x @ wv + bv
    ew = jnp.exp(wbias[:t, :t])
    ek = jnp.exp(k)
    num = jnp.einsum('ts,bsh->bth', ew, ek * v)
    den = jnp.einsum('ts,bsh->bth', ew, ek)
    y = jax.nn.sigmoid(q) * (num / den)
    return y @ wp + bp
