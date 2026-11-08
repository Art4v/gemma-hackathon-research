"""MoE expert bincount, group offsets, and capacity-limited float-scaled dispatch mask."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_expert_bincount_dispatch_mask',
    'num_tokens': 1048576,
    'top_k': 8,
    'num_experts': 16,
    'capacity': 128,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    T = CONFIG['num_tokens']
    K = CONFIG['top_k']
    E = CONFIG['num_experts']
    
    # Activations: routing probabilities from top-k softmax
    gate_probs = jax.random.uniform(k1, (T, K), dtype=dtype, minval=0.01, maxval=1.0)
    # Integer inputs: selected expert indices per token
    expert_indices = jax.random.randint(k2, (T, K), minval=0, maxval=E, dtype=jnp.int32)
    # Parameters: per-expert scaling factor
    expert_scale = jnp.ones((E,), dtype=dtype)
    
    return gate_probs, expert_indices, expert_scale

def workload(gate_probs, expert_indices, expert_scale):
    T, K = expert_indices.shape
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    eps = CONFIG['eps']
    
    # Normalize top-k gate probabilities in float32
    gates_f32 = gate_probs.astype(jnp.float32)
    gates_norm = gates_f32 / (jnp.sum(gates_f32, axis=-1, keepdims=True) + eps)
    
    # One-hot encode expert assignments: (T, K, E) in int32
    one_hot = lax.broadcasted_iota(jnp.int32, (T, K, E), 2) == jnp.expand_dims(expert_indices, -1)
    one_hot_i32 = one_hot.astype(jnp.int32)
    
    # Flatten across (T * K) routing decisions to assign sequential slots per expert
    flat_oh = one_hot_i32.reshape(T * K, E)
    
    # Per-expert token counts and exclusive group offsets
    expert_counts = jnp.sum(flat_oh, axis=0, dtype=jnp.int32)
    group_offsets = jnp.cumsum(expert_counts, dtype=jnp.int32) - expert_counts
    
    # Exclusive prefix sum along flattened tokens gives 0-indexed position in expert buffer
    flat_pos = jnp.cumsum(flat_oh, axis=0, dtype=jnp.int32) - flat_oh
    pos_in_expert = flat_pos.reshape(T, K, E)
    
    # Apply capacity limit: keep only tokens where assigned slot < C
    valid_mask = one_hot & (pos_in_expert < C)
    
    # Scale by normalized gate weights and expert scale
    scaled_gates = jnp.expand_dims(gates_norm, -1) * expert_scale.astype(jnp.float32).reshape(1, 1, E)
    dispatch_weights = jnp.where(valid_mask, scaled_gates, 0.0)
    
    # Sum over top-k dimension to get per-token, per-expert dispatch weight and target slot
    # (In standard top-k routing, a token routes to distinct experts so at most one k is active per e)
    token_expert_weights = jnp.sum(dispatch_weights, axis=1).astype(gate_probs.dtype)
    token_expert_slots = jnp.sum(jnp.where(valid_mask, pos_in_expert, -1), axis=1).astype(jnp.int32)
    
    return expert_counts, group_offsets, token_expert_weights, token_expert_slots
