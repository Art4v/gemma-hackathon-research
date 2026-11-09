"""SAB: MAB(X, X) = multihead self-attention with residual Q, then O + relu(fc_o(O)) (ln=False)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SAB_4902',
    'batch': 8,
    'seq_len': 2048,
    'dim_in': 1024,
    'dim_out': 1024,
    'num_heads': 4,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    b, n = CONFIG['batch'], CONFIG['seq_len']
    din, dout = CONFIG['dim_in'], CONFIG['dim_out']
    x = jax.random.uniform(keys[0], (b, n, din), dtype)
    wq = jax.random.normal(keys[1], (din, dout), dtype) * 0.02
    bq = jax.random.normal(keys[2], (dout,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (din, dout), dtype) * 0.02
    bk = jax.random.normal(keys[4], (dout,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (din, dout), dtype) * 0.02
    bv = jax.random.normal(keys[6], (dout,), dtype) * 0.02
    wo = jax.random.normal(keys[7], (dout, dout), dtype) * 0.02
    bo = jax.random.normal(keys[8], (dout,), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv, wo, bo


def workload(x, wq, bq, wk, bk, wv, bv, wo, bo):
    b, n, _ = x.shape
    dim_v = wq.shape[1]
    h = CONFIG['num_heads']
    d = dim_v // h
    q = (x @ wq + bq).reshape(b, n, h, d)
    k = (x @ wk + bk).reshape(b, n, h, d)
    v = (x @ wv + bv).reshape(b, n, h, d)
    # Scale by sqrt(dim_V), not sqrt(head_dim), matching the original module.
    s = jnp.einsum('bqhd,bkhd->bhqk', q, k) / (dim_v ** 0.5)
    a = jax.nn.softmax(s, axis=-1)
    o = q + jnp.einsum('bhqk,bkhd->bqhd', a, v)
    o = o.reshape(b, n, dim_v)
    return o + jax.nn.relu(o @ wo + bo)
