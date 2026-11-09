"""[bf16 variant] Conformer relative-position multi-head self-attention: QKV/pos projections, biased content+position scores, masked softmax, attention, output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'conformer_relpos_mhsa',
    'batch': 8,
    'seq_len': 1024,
    'n_feat': 512,
    'n_head': 8,
}


def create_inputs(dtype=jnp.bfloat16):
    b, t, d, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['n_feat'], CONFIG['n_head']
    dk = d // h
    keys = jax.random.split(jax.random.key(0), 16)
    query = jax.random.uniform(keys[0], (b, t, d), dtype)
    key = jax.random.uniform(keys[1], (b, t, d), dtype)
    value = jax.random.uniform(keys[2], (b, t, d), dtype)
    pos_emb = jax.random.uniform(keys[3], (b, t, d), dtype)
    mask = jax.random.randint(keys[4], (b, t, t), 0, 2).astype(dtype)
    wq = jax.random.normal(keys[5], (d, d), dtype) * 0.02
    bq = jax.random.normal(keys[6], (d,), dtype) * 0.02
    wk = jax.random.normal(keys[7], (d, d), dtype) * 0.02
    bk = jax.random.normal(keys[8], (d,), dtype) * 0.02
    wv = jax.random.normal(keys[9], (d, d), dtype) * 0.02
    bv = jax.random.normal(keys[10], (d,), dtype) * 0.02
    wo = jax.random.normal(keys[11], (d, d), dtype) * 0.02
    bo = jax.random.normal(keys[12], (d,), dtype) * 0.02
    wpos = jax.random.normal(keys[13], (d, d), dtype) * 0.02
    pos_bias_u = jax.random.normal(keys[14], (h, dk), dtype) * 0.02
    pos_bias_v = jax.random.normal(keys[15], (h, dk), dtype) * 0.02
    return (query, key, value, pos_emb, mask, wq, bq, wk, bk, wv, bv, wo, bo,
            wpos, pos_bias_u, pos_bias_v)


def workload(query, key, value, pos_emb, mask, wq, bq, wk, bk, wv, bv, wo, bo,
             wpos, pos_bias_u, pos_bias_v):
    n_batch = query.shape[0]
    n_head, d_k = pos_bias_u.shape
    # (B, T, H, d_k) layout, as in torch after q.transpose(1, 2)
    q = (query @ wq + bq).reshape(n_batch, -1, n_head, d_k)
    k = (key @ wk + bk).reshape(n_batch, -1, n_head, d_k).transpose(0, 2, 1, 3)
    v = (value @ wv + bv).reshape(n_batch, -1, n_head, d_k).transpose(0, 2, 1, 3)
    p = (pos_emb @ wpos).reshape(pos_emb.shape[0], -1, n_head, d_k).transpose(0, 2, 1, 3)

    q_u = (q + pos_bias_u).transpose(0, 2, 1, 3)
    q_v = (q + pos_bias_v).transpose(0, 2, 1, 3)
    matrix_ac = jnp.matmul(q_u, jnp.swapaxes(k, -2, -1))
    matrix_bd = jnp.matmul(q_v, jnp.swapaxes(p, -2, -1))
    scores = (matrix_ac + matrix_bd) / jnp.sqrt(jnp.asarray(d_k, scores_dtype := query.dtype))

    masked = (mask[:, None] == 0)
    scores = jnp.where(masked, -jnp.inf, scores)
    attn = jnp.where(masked, 0.0, jax.nn.softmax(scores, axis=-1)).astype(scores.dtype)

    x = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(n_batch, -1, n_head * d_k)
    return x @ wo + bo
