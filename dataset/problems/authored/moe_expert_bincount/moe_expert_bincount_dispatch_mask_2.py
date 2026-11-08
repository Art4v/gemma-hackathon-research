"""MoE expert bincount, group offsets, and capacity-limited gate mask computation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_expert_bincount_dispatch_mask',
    'num_tokens': 262144,
    'num_experts': 128,
    'top_k': 8,
    'expert_capacity': 20480,
    'routed_scaling_factor': 2.5,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    num_tokens = CONFIG['num_tokens']
    num_experts = CONFIG['num_experts']
    top_k = CONFIG['top_k']

    router_logits = jax.random.normal(k1, (num_tokens, num_experts), dtype=dtype)
    raw_gates = jax.random.uniform(k2, (num_tokens, top_k), dtype=dtype, minval=0.01, maxval=1.0)
    expert_indices = jax.random.randint(k3, (num_tokens, top_k), minval=0, maxval=num_experts, dtype=jnp.int32)
    expert_bias = jnp.zeros((num_experts,), dtype=dtype)

    return router_logits, raw_gates, expert_indices, expert_bias


def workload(router_logits, raw_gates, expert_indices, expert_bias):
    num_tokens = CONFIG['num_tokens']
    num_experts = CONFIG['num_experts']
    top_k = CONFIG['top_k']
    capacity = CONFIG['expert_capacity']
    scale = CONFIG['routed_scaling_factor']
    eps = CONFIG['eps']

    logits_f32 = router_logits.astype(jnp.float32) + expert_bias.astype(jnp.float32)[None, :]
    router_probs = jax.nn.sigmoid(logits_f32)
    gathered_probs = jnp.take_along_axis(router_probs, expert_indices, axis=-1)

    combined_gates = gathered_probs * raw_gates.astype(jnp.float32)
    gate_norm = jnp.sum(combined_gates, axis=-1, keepdims=True) + eps
    norm_gates = (combined_gates / gate_norm) * scale

    one_hot = jax.nn.one_hot(expert_indices, num_experts, dtype=jnp.int32)
    token_expert_counts = jnp.sum(one_hot, axis=1)
    expert_counts = jnp.sum(token_expert_counts, axis=0)

    group_offsets = jnp.cumsum(expert_counts) - expert_counts

    prefix_per_token = jnp.cumsum(token_expert_counts, axis=0) - token_expert_counts
    base_ranks = jnp.take_along_axis(prefix_per_token, expert_indices, axis=1)

    intra_token_offset = jnp.zeros_like(expert_indices)
    for k in range(1, top_k):
        matches = jnp.sum((expert_indices[:, :k] == expert_indices[:, k:k+1]).astype(jnp.int32), axis=1)
        intra_token_offset = intra_token_offset.at[:, k].set(matches)

    position_in_expert = base_ranks + intra_token_offset
    valid_mask = position_in_expert < capacity

    masked_gates = jnp.where(valid_mask, norm_gates, 0.0).astype(raw_gates.dtype)
    dispatch_destinations = jnp.where(
        valid_mask,
        expert_indices * capacity + position_in_expert,
        -1
    )
    clipped_counts = jnp.minimum(expert_counts, capacity)

    return expert_counts, clipped_counts, group_offsets, dispatch_destinations, masked_gates
