"""T5 bucketed relative position bias -> (QK^T * scale + bias) -> softmax -> @V (bidirectional encoder self-attention)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 't5_relative_bias_attention',
    'batch': 6,
    'num_heads': 16,
    'seq_len': 2048,
    'head_dim': 64,
    'num_buckets': 32,
    'max_distance': 128,
    'bidirectional': True,
    'scale': 0.125,  # 1/sqrt(head_dim); T5 folds this into init, kept explicit for numerical sanity
    'bias_std': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    B, H, S, D = CONFIG['batch'], CONFIG['num_heads'], CONFIG['seq_len'], CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, kb = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, S, D), dtype)
    k = jax.random.normal(kk, (B, H, S, D), dtype)
    v = jax.random.normal(kv, (B, H, S, D), dtype)
    bias_table = (jax.random.normal(kb, (CONFIG['num_buckets'], H), jnp.float32)
                  * CONFIG['bias_std']).astype(dtype)
    return q, k, v, bias_table


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    ret = jnp.zeros_like(relative_position)
    if bidirectional:
        num_buckets = num_buckets // 2
        ret = ret + (relative_position > 0).astype(jnp.int32) * num_buckets
        n = jnp.abs(relative_position)
    else:
        n = jnp.maximum(-relative_position, 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    n_safe = jnp.maximum(n, 1).astype(jnp.float32)
    val_if_large = max_exact + (
        jnp.log(n_safe / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    return ret + jnp.where(is_small, n, val_if_large)


def workload(q, k, v, bias_table):
    Sq = q.shape[2]
    Sk = k.shape[2]
    q_pos = jnp.arange(Sq, dtype=jnp.int32)[:, None]
    k_pos = jnp.arange(Sk, dtype=jnp.int32)[None, :]
    buckets = _relative_position_bucket(
        k_pos - q_pos, CONFIG['bidirectional'], CONFIG['num_buckets'], CONFIG['max_distance'])
    # [Sq, Sk, H] -> [H, Sq, Sk]
    bias = jnp.transpose(bias_table.astype(jnp.float32)[buckets], (2, 0, 1))

    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    vf = v.astype(jnp.float32)
    scores = jnp.einsum('bhqd,bhkd->bhqk', qf, kf) * CONFIG['scale']
    scores = scores + bias[None]
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', probs, vf)
    return out.astype(q.dtype)
