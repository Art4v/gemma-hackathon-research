"""ISAB: two multi-head attention blocks (inducing points attend to x, then x attends to result), each with residual + relu(fc) residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ISAB_8416',
    'batch': 16,
    'seq_len': 4096,
    'd': 512,
    'n_heads': 4,
    'n_inducing_points': 128,
}


def create_inputs(dtype=jnp.float32):
    B, N, d, P = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d'], CONFIG['n_inducing_points']
    keys = jax.random.split(jax.random.key(0), 22)
    x = jax.random.uniform(keys[0], (B, N, d), dtype)
    I = jax.random.normal(keys[1], (1, P, d), dtype) * 0.02
    params = []
    k = 2
    for _ in range(2):  # mab0, mab1
        for _ in range(5):  # q, k, v, o projectors, fc
            params.append(jax.random.normal(keys[k], (d, d), dtype) * 0.02)
            params.append(jax.random.normal(keys[k + 1], (d,), dtype) * 0.02)
            k += 2
    return (x, I, *params)


def _mha(q_in, kv_in, wq, bq, wk, bk, wv, bv, wo, bo):
    H = CONFIG['n_heads']
    B, N, _ = q_in.shape
    M = kv_in.shape[1]
    q = (q_in @ wq + bq).reshape(B, N, H, -1)
    k = (kv_in @ wk + bk).reshape(B, M, H, -1)
    v = (kv_in @ wv + bv).reshape(B, M, H, -1)
    scale = jnp.sqrt(jnp.asarray(q.shape[-1], q.dtype))
    s = jnp.einsum('bnhd,bmhd->bhnm', q, k) / scale
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('bhnm,bmhd->bnhd', p, v).reshape(B, N, -1)
    return o @ wo + bo


def _mab(queries, keys, params):
    h = _mha(queries, keys, *params[:8]) + queries
    fc_w, fc_b = params[8], params[9]
    return h + jax.nn.relu(h @ fc_w + fc_b)


def workload(x, I, *params):
    B = x.shape[0]
    inducing = jnp.broadcast_to(I, (B,) + I.shape[1:])
    h = _mab(inducing, x, params[:10])
    return _mab(x, h, params[10:20])
