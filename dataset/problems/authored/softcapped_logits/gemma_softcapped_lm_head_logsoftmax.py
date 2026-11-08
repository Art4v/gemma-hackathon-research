"""Gemma final RMSNorm -> tied unembedding matmul -> tanh logit soft-capping -> log-softmax (bf16 in, f32 out)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma_softcapped_lm_head_logsoftmax',
    'num_tokens': 256,
    'hidden_dim': 1152,
    'vocab_size': 262144,
    'rms_eps': 1e-6,
    'final_logit_softcap': 30.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_h, k_w = jax.random.split(key, 2)
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    V = CONFIG['vocab_size']
    hidden = jax.random.normal(k_h, (T, D), dtype)
    embedding = jax.random.normal(k_w, (V, D), dtype) * CONFIG['weight_scale']
    embedding = embedding.astype(dtype)
    norm_weight = jnp.zeros((D,), dtype)  # Gemma RMSNorm uses (1 + w)
    return hidden, embedding, norm_weight


def workload(hidden, embedding, norm_weight):
    eps = CONFIG['rms_eps']
    cap = CONFIG['final_logit_softcap']
    in_dtype = hidden.dtype

    # Gemma RMSNorm: compute in f32, scale by (1 + w), cast back to activation dtype.
    h = hidden.astype(jnp.float32)
    var = jnp.mean(h * h, axis=-1, keepdims=True)
    h = h * lax.rsqrt(var + eps)
    h = h * (1.0 + norm_weight.astype(jnp.float32))
    h = h.astype(in_dtype)

    # Tied unembedding: [T, D] x [V, D]^T -> [T, V], f32 accumulation.
    logits = lax.dot_general(
        h, embedding,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    # Final logit soft-capping.
    logits = cap * jnp.tanh(logits / cap)

    # Full-vocab log-softmax in f32.
    m = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return shifted - lse
