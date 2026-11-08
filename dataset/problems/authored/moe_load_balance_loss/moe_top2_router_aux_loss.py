"""MoE router: linear projection -> softmax over experts -> top-2 gates (renormalized) + Switch-style auxiliary load-balancing loss."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_top2_router_aux_loss',
    'batch': 8,
    'seq_len': 4096,
    'd_model': 4096,
    'num_experts': 64,
    'top_k': 2,
    'aux_loss_coef': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key, 2)
    num_tokens = CONFIG['batch'] * CONFIG['seq_len']
    x = jax.random.normal(k_x, (num_tokens, CONFIG['d_model']), dtype=dtype)
    w_router = jax.random.normal(
        k_w, (CONFIG['d_model'], CONFIG['num_experts']), dtype=dtype) * 0.02
    return x, w_router


def workload(x, w_router):
    top_k = CONFIG['top_k']
    num_tokens = x.shape[0]
    num_experts = w_router.shape[1]

    # Router logits accumulated in float32.
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)  # [T, E] float32

    # Top-k selection and renormalization of the selected gates.
    top_vals, top_idx = lax.top_k(probs, top_k)  # [T, k]
    gates = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    # Switch-style load-balancing loss.
    dispatch = jax.nn.one_hot(top_idx, num_experts, dtype=jnp.float32)  # [T, k, E]
    tokens_per_expert = jnp.sum(dispatch, axis=(0, 1)) / (num_tokens * top_k)  # f_e
    mean_prob_per_expert = jnp.mean(probs, axis=0)  # P_e
    aux_loss = (CONFIG['aux_loss_coef'] * num_experts *
                jnp.sum(tokens_per_expert * mean_prob_per_expert))

    return gates.astype(x.dtype), top_idx.astype(jnp.int32), aux_loss.astype(jnp.float32)
