"""Masked self-attention: relu(Linear) V/Q/K -> masked bmm softmax -> bmm -> relu(Linear)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttModel_13790',
    'batch': 16,
    'seq_len': 1024,
    'din': 1024,
    'hidden_dim': 1024,
    'dout': 1024,
    'mask_penalty': 9000000000000000.0,
}


def create_inputs(dtype=jnp.float32):
    b, n = CONFIG['batch'], CONFIG['seq_len']
    din, h, dout = CONFIG['din'], CONFIG['hidden_dim'], CONFIG['dout']
    ks = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(ks[0], (b, n, din), dtype)
    mask = jax.random.uniform(ks[1], (b, n, n), dtype)
    wv = jax.random.normal(ks[2], (din, h), dtype) * 0.02
    bv = jax.random.normal(ks[3], (h,), dtype) * 0.02
    wk = jax.random.normal(ks[4], (din, h), dtype) * 0.02
    bk = jax.random.normal(ks[5], (h,), dtype) * 0.02
    wq = jax.random.normal(ks[6], (din, h), dtype) * 0.02
    bq = jax.random.normal(ks[7], (h,), dtype) * 0.02
    wout = jax.random.normal(ks[8], (h, dout), dtype) * 0.02
    bout = jax.random.normal(ks[9], (dout,), dtype) * 0.02
    return x, mask, wv, bv, wk, bk, wq, bq, wout, bout


def workload(x, mask, wv, bv, wk, bk, wq, bq, wout, bout):
    v = jax.nn.relu(x @ wv + bv)
    q = jax.nn.relu(x @ wq + bq)
    k = jax.nn.relu(x @ wk + bk)
    scores = jnp.einsum('bnh,bmh->bnm', q, k)
    logits = scores * mask - CONFIG['mask_penalty'] * (1 - mask)
    att = jax.nn.softmax(logits, axis=2)
    out = jnp.einsum('bnm,bmh->bnh', att, v)
    return jax.nn.relu(out @ wout + bout)
