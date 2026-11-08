"""Chunked prefill GQA causal attention with per-batch prefix offsets and logit soft-capping."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'chunked_prefill_gqa_attention',
    'batch': 8,
    'q_heads': 32,
    'kv_heads': 8,
    'chunk_len': 512,
    'max_kv_len': 8192,
    'head_dim': 128,
    'softcap': 50.0,
    'min_prefix': 1024,
    'max_prefix': 7680,
}


def create_inputs(dtype=jnp.bfloat16):
    k_q, k_k, k_v, k_len = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch']
    Hq = CONFIG['q_heads']
    Hkv = CONFIG['kv_heads']
    C = CONFIG['chunk_len']
    S = CONFIG['max_kv_len']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, Hq, C, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, Hkv, S, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, Hkv, S, D), dtype=dtype)
    prefix_lens = jax.random.randint(
        k_len, (B,), CONFIG['min_prefix'], CONFIG['max_prefix'] + 1, dtype=jnp.int32
    )
    return q, k, v, prefix_lens


def workload(q, k, v, prefix_lens):
    B, Hq, C, D = q.shape
    _, Hkv, S, _ = k.shape
    groups = Hq // Hkv
    scale = D ** -0.5
    softcap = CONFIG['softcap']

    q_grouped = q.reshape(B, Hkv, groups, C, D).astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.einsum('bhgcd,bhsd->bhgcs', q_grouped, k_f32) * scale
    logits = softcap * jnp.tanh(logits / softcap)

    q_pos = prefix_lens[:, None, None] + jnp.arange(C, dtype=jnp.int32)[None, :, None]
    k_pos = jnp.arange(S, dtype=jnp.int32)[None, None, :]
    mask = k_pos <= q_pos
    mask = mask[:, None, None, :, :]

    logits = jnp.where(mask, logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhgcs,bhsd->bhgcd', weights, v_f32)
    return out.reshape(B, Hq, C, D).astype(q.dtype)
