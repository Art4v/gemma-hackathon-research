"""MoE Top-2 Router with Softmax, Gate Normalization, and Switch Auxiliary Load-Balancing Loss."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_top2_router_load_balance_loss',
    'batch_size': 16,
    'seq_len': 4096,
    'hidden_size': 4096,
    'num_experts': 64,
    'top_k': 2,
    'aux_loss_alpha': 0.01,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    D = CONFIG['hidden_size']
    E = CONFIG['num_experts']
    hidden_states = jax.random.normal(k1, (B, S, D), dtype=dtype)
    router_weights = jax.random.normal(k2, (D, E), dtype=dtype) * 0.02
    return (hidden_states, router_weights)

def workload(hidden_states, router_weights):
    B, S, D = hidden_states.shape
    E = CONFIG['num_experts']
    K = CONFIG['top_k']
    alpha = CONFIG['aux_loss_alpha']
    eps = CONFIG['eps']
    num_tokens = B * S

    tokens = jnp.reshape(hidden_states, (num_tokens, D))
    logits = jnp.dot(tokens, router_weights, preferred_element_type=jnp.float32)

    probs = jax.nn.softmax(logits, axis=-1)

    top_k_gates, top_k_indices = lax.top_k(probs, K)
    gate_sum = jnp.sum(top_k_gates, axis=-1, keepdims=True)
    norm_gates = (top_k_gates / (gate_sum + eps)).astype(hidden_states.dtype)

    prob_mean_per_expert = jnp.mean(probs, axis=0)

    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=E, dtype=jnp.float32)
    token_assignment_per_expert = jnp.sum(expert_mask, axis=1)
    fraction_per_expert = jnp.mean(token_assignment_per_expert, axis=0)

    aux_loss = alpha * E * jnp.sum(prob_mean_per_expert * fraction_per_expert)

    norm_gates_out = jnp.reshape(norm_gates, (B, S, K))
    top_k_indices_out = jnp.reshape(top_k_indices, (B, S, K))
    return norm_gates_out, top_k_indices_out, aux_loss
