"""[bf16 variant] Hedgehog feature map (softmax([2(xW+b), -2(xW+b)])) on q,k -> causal chunked linear attention with normalization."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'hedgehog_linear_attention',
    'batch': 8,
    'seq_len': 4096,
    'num_heads': 8,
    'head_dim': 128,
    'chunk_size': 64,
    'scale': 1.0,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    b, t, h, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    kq, kk, kv, kwq, kwk = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(kq, (b, t, h, d), dtype)
    k = jax.random.normal(kk, (b, t, h, d), dtype)
    v = jax.random.normal(kv, (b, t, h, d), dtype)
    # ponytail: torch inits these to identity; small noise keeps the map near init but non-trivial.
    wq = jnp.eye(d, dtype=dtype) + jax.random.normal(kwq, (d, d), dtype) * 0.02
    wk = jnp.eye(d, dtype=dtype) + jax.random.normal(kwk, (d, d), dtype) * 0.02
    bq = jnp.zeros((d,), dtype)
    bk = jnp.zeros((d,), dtype)
    return q, k, v, wq, bq, wk, bk


def _feature_map(x, w, b):
    y = 2.0 * (x @ w + b)
    return jax.nn.softmax(jnp.concatenate([y, -y], axis=-1), axis=-1)


def workload(q, k, v, wq, bq, wk, bk):
    c = CONFIG['chunk_size']
    q = _feature_map(q, wq, bq) * CONFIG['scale']
    k = _feature_map(k, wk, bk)
    b, l, h, _ = q.shape
    n = l // c

    def to_chunks(x):
        return x.reshape(b, n, c, h, x.shape[-1]).transpose(0, 3, 1, 2, 4)

    qc, kc, vc = to_chunks(q), to_chunks(k), to_chunks(v)
    kv = jnp.cumsum(jnp.swapaxes(kc, -1, -2) @ vc, axis=2)
    kv = jnp.concatenate([jnp.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], axis=2)
    inter = qc @ kv
    mask = jnp.tril(jnp.ones((c, c), dtype=bool))
    intra = jnp.where(mask, qc @ jnp.swapaxes(kc, -1, -2), 0.0) @ vc
    o = (inter + intra).transpose(0, 2, 3, 1, 4).reshape(b, l, h, -1)
    denom = jnp.sum(q * jnp.cumsum(k, axis=1), axis=-1, keepdims=True) + CONFIG['eps']
    return o / denom
