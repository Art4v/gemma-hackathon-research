"""Fused linear projection, logit softcapping, and per-token cross-entropy loss (Gemma-2 style)."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma2_2b_fused_linear_softcap_cross_entropy',
    'num_tokens': 8192,
    'hidden_size': 2304,
    'vocab_size': 16384,
    'softcap': 30.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_hid, k_weight, k_target = jax.random.split(jax.random.key(0), 3)
    num_tokens = CONFIG['num_tokens']
    hidden_size = CONFIG['hidden_size']
    vocab_size = CONFIG['vocab_size']

    hidden_states = jax.random.normal(k_hid, (num_tokens, hidden_size), dtype=dtype)
    lm_head_weight = jax.random.normal(k_weight, (hidden_size, vocab_size), dtype=dtype) * 0.02
    targets = jax.random.randint(k_target, (num_tokens,), 0, vocab_size, dtype=jnp.int32)
    return hidden_states, targets, lm_head_weight


def workload(hidden_states, targets, lm_head_weight):
    softcap = CONFIG['softcap']
    num_tokens = hidden_states.shape[0]

    logits = jnp.dot(hidden_states, lm_head_weight, preferred_element_type=jnp.float32)
    capped_logits = jnp.tanh(logits / softcap) * softcap

    max_logits = jnp.max(capped_logits, axis=-1, keepdims=True)
    shifted_logits = capped_logits - max_logits
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1)) + jnp.squeeze(max_logits, axis=-1)

    target_logits = capped_logits[jnp.arange(num_tokens), targets]
    per_token_loss = log_sum_exp - target_logits
    return per_token_loss
