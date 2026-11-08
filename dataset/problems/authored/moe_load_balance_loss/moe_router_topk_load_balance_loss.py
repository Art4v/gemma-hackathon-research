"""MoE router projection with softmax, top-2 selection, gate normalization, and Switch load-balancing auxiliary loss."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    "name": "moe_router_topk_load_balance_loss",
    "batch_size": 8,
    "seq_len": 4096,
    "hidden_size": 4096,
    "num_experts": 64,
    "top_k": 2,
    "aux_loss_alpha": 0.01,
    "eps": 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    hidden_states = jax.random.normal(
        k1,
        (CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["hidden_size"]),
        dtype=dtype,
    )
    router_weights = (
        jax.random.normal(
            k2,
            (CONFIG["hidden_size"], CONFIG["num_experts"]),
            dtype=dtype,
        )
        * 0.02
    )
    return (hidden_states, router_weights)


def workload(hidden_states, router_weights):
    bsz, seq_len, hidden_size = hidden_states.shape
    num_experts = CONFIG["num_experts"]
    top_k = CONFIG["top_k"]
    num_tokens = bsz * seq_len

    flat_hidden = jnp.reshape(hidden_states, (num_tokens, hidden_size))
    logits = jnp.dot(flat_hidden, router_weights, preferred_element_type=jnp.float32)

    probs = jax.nn.softmax(logits, axis=-1)
    topk_weights, topk_indices = lax.top_k(probs, top_k)

    norm_denom = jnp.sum(topk_weights, axis=-1, keepdims=True) + CONFIG["eps"]
    routing_weights = (topk_weights / norm_denom).astype(hidden_states.dtype)

    expert_mask = jax.nn.one_hot(topk_indices, num_classes=num_experts, dtype=jnp.float32)
    token_fraction = jnp.mean(jnp.sum(expert_mask, axis=1), axis=0)
    router_prob_per_expert = jnp.mean(probs, axis=0)

    aux_loss = (
        CONFIG["aux_loss_alpha"]
        * num_experts
        * jnp.sum(token_fraction * router_prob_per_expert)
    )

    routing_weights = jnp.reshape(routing_weights, (bsz, seq_len, top_k))
    topk_indices = jnp.reshape(topk_indices, (bsz, seq_len, top_k))
    return routing_weights, topk_indices, aux_loss
