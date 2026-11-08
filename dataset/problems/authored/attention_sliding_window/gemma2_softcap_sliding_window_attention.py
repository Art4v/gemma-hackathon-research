"""Gemma-2 / GPT-OSS style sliding-window GQA causal attention with logit softcapping and attention sinks."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma2_softcap_sliding_window_attention',
    'batch_size': 2,
    'seq_len': 8192,
    'num_q_heads': 16,
    'num_kv_heads': 4,
    'head_dim': 128,
    'window_size': 512,
    'softcap': 50.0,
    'q_block_size': 256,
    'mask_value': -1e30,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    T = CONFIG['seq_len']
    H_Q = CONFIG['num_q_heads']
    H_KV = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, T, H_Q, D), dtype=dtype)
    k = jax.random.normal(k2, (B, T, H_KV, D), dtype=dtype)
    v = jax.random.normal(k3, (B, T, H_KV, D), dtype=dtype)
    sinks = jax.random.normal(k4, (H_Q,), dtype=jnp.float32) * 0.02
    return q, k, v, sinks

def workload(q, k, v, sinks):
    B, T, H_Q, D = q.shape
    H_KV = k.shape[2]
    G = H_Q // H_KV
    W = CONFIG['window_size']
    softcap = CONFIG['softcap']
    q_block = CONFIG['q_block_size']
    mask_val = CONFIG['mask_value']
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    q_grouped = jnp.reshape(q, (B, T, H_KV, G, D))
    sinks_grouped = jnp.reshape(sinks.astype(jnp.float32), (H_KV, G, 1, 1))

    num_blocks = T // q_block
    q_blocks = jnp.reshape(q_grouped, (B, num_blocks, q_block, H_KV, G, D))
    q_blocks = jnp.transpose(q_blocks, (1, 0, 3, 4, 2, 5))

    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    k_idx = jnp.arange(T)[None, :]

    def step(carry, inputs):
        block_idx, q_blk = inputs
        q_idx = (block_idx * q_block + jnp.arange(q_block))[:, None]

        logits = jnp.einsum('bkhqd,bktd->bkhqt', q_blk.astype(jnp.float32), k_t.astype(jnp.float32))
        logits = logits * scale
        logits = softcap * jnp.tanh(logits / softcap)

        diff = q_idx - k_idx
        in_window = (diff >= 0) & (diff < W)
        logits = jnp.where(in_window[None, None, None, :, :], logits, mask_val)

        sink_expanded = jnp.broadcast_to(sinks_grouped[None, :, :, :, :], (B, H_KV, G, q_block, 1))
        logits_with_sink = jnp.concatenate([logits, sink_expanded], axis=-1)

        max_logits = jnp.max(logits_with_sink, axis=-1, keepdims=True)
        exp_logits = jnp.exp(logits_with_sink - max_logits)
        denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits[..., :-1] / denom

        out_blk = jnp.einsum('bkhqt,bktd->bkhqd', probs, v_t.astype(jnp.float32))
        return carry, out_blk.astype(q.dtype)

    _, out_blocks = lax.scan(step, None, (jnp.arange(num_blocks), q_blocks))
    out = jnp.transpose(out_blocks, (1, 0, 4, 2, 3, 5))
    return jnp.reshape(out, (B, T, H_Q, D))
