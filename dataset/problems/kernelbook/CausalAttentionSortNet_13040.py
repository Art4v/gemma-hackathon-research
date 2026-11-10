"""[bf16 variant] CausalAttentionSortNet: cumavg -> bucket -> sq.sk^T scale -> causal mask -> differentiable top-k (softmax/argmax/scatter)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CausalAttentionSortNet_13040',
    'batch_heads': 32,
    'seq_len': 4096,
    'kv_seq_len': 4096,
    'head_dim': 128,
    # Module hyperparameters, kept equal to get_init_inputs() so the torch check matches.
    'heads': 4,
    'bucket_size': 4,
    'dim': 4,
    'temperature': 4.0,
    'topk': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kq, kk = jax.random.split(jax.random.key(0))
    q = jax.random.uniform(kq, (CONFIG['batch_heads'], CONFIG['seq_len'], CONFIG['head_dim']), dtype)
    k = jax.random.uniform(kk, (CONFIG['batch_heads'], CONFIG['kv_seq_len'], CONFIG['head_dim']), dtype)
    return q, k


def _cumavg(t):
    r = jnp.arange(1, t.shape[1] + 1, dtype=t.dtype)
    return jnp.cumsum(t, axis=1) / r[None, :, None]


def _differentiable_topk(x, k, temperature):
    *lead, n, cols = x.shape
    outs = []
    for i in range(k):
        probs = jax.nn.softmax(x / temperature, axis=-1)
        idx = jnp.argmax(probs, axis=-1)
        vals = jnp.max(probs, axis=-1, keepdims=True)
        onehot = jax.nn.one_hot(idx, cols, dtype=jnp.bool_)
        outs.append(jnp.where(onehot, vals, jnp.zeros_like(x)))
        if i != k - 1:
            x = jnp.where(onehot, -jnp.inf, x)
    topks = jnp.concatenate(outs, axis=-1)
    return topks.reshape(*lead, k * n, cols)


def workload(q, k):
    bucket_size = CONFIG['bucket_size']
    topk = CONFIG['topk']
    bh, n, d = q.shape
    buckets = n // bucket_size
    kv_buckets = k.shape[1] // bucket_size

    q_r = _cumavg(q).reshape(bh, buckets, bucket_size, d)
    k_r = _cumavg(k).reshape(bh, kv_buckets, bucket_size, k.shape[-1])
    sq = q_r[:, :, 0]
    sk = jnp.sum(k_r, axis=2)
    sk = jnp.pad(sk, ((0, 0), (topk, 0), (0, 0)))

    R = jnp.einsum('bie,bje->bij', sq, sk, precision=lax.Precision.HIGHEST) * (CONFIG['dim'] ** -0.5)

    rows = jnp.arange(R.shape[1])[:, None]
    cols = jnp.arange(R.shape[2])[None, :]
    mask = (cols >= rows + topk) & (cols - topk < buckets)
    R = jnp.where(mask[None], -jnp.finfo(R.dtype).max, R)
    return _differentiable_topk(R, topk, CONFIG['temperature'])
