"""MoE router top-k selection, per-expert token bincount, group offsets, and capacity-limited float-scaled dispatch mask."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_expert_bincount_dispatch_mask',
    'num_tokens': 131072,
    'num_experts': 256,
    'top_k': 4,
    'capacity': 2560,
    'routed_scaling_factor': 2.5,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    T = CONFIG['num_tokens']
    E = CONFIG['num_experts']
    router_logits = jax.random.normal(k1, (T, E), dtype=dtype)
    expert_bias = jax.random.normal(k2, (E,), dtype=dtype) * 0.02
    return (router_logits, expert_bias)


def workload(router_logits, expert_bias):
    T = CONFIG['num_tokens']
    E = CONFIG['num_experts']
    K = CONFIG['top_k']
    C = CONFIG['capacity']
    scale = CONFIG['routed_scaling_factor']
    eps = CONFIG['eps']

    logits_f32 = router_logits.astype(jnp.float32)
    bias_f32 = expert_bias.astype(jnp.float32)

    scores = jax.nn.sigmoid(logits_f32)
    biased_scores = scores + bias_f32[None, :]

    _, topk_indices = lax.top_k(biased_scores, K)
    topk_gates = jnp.take_along_axis(scores, topk_indices, axis=-1)
    gate_sum = jnp.sum(topk_gates, axis=-1, keepdims=True) + eps
    norm_gates = (topk_gates / gate_sum) * scale

    expert_ids = jnp.arange(E, dtype=jnp.int32)
    matches = (topk_indices[:, :, None] == expert_ids[None, None, :])
    token_expert_match = jnp.any(matches, axis=1)
    token_expert_gate = jnp.sum(norm_gates[:, :, None] * matches.astype(jnp.float32), axis=1)

    expert_counts = jnp.sum(token_expert_match.astype(jnp.int32), axis=0, dtype=jnp.int32)
    clamped_counts = jnp.minimum(expert_counts, C)
    group_offsets = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.cumsum(clamped_counts, axis=0, dtype=jnp.int32)
    ], axis=0)

    positions = jnp.cumsum(token_expert_match.astype(jnp.int32), axis=0, dtype=jnp.int32) - 1
    valid_mask = token_expert_match & (positions < C)

    dispatch_weights = jnp.where(valid_mask, token_expert_gate, 0.0).astype(router_logits.dtype)
    dispatch_slots = jnp.where(valid_mask, positions, -1).astype(jnp.int32)

    return expert_counts, group_offsets, dispatch_weights, dispatch_slots
