"""MoE router projection with softmax, top-2 expert selection, and Switch-style auxiliary load-balancing loss."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_top2_router_load_balance_loss',
    'batch': 8,
    'seq_len': 4096,
    'hidden_dim': 4096,
    'num_experts': 64,
    'top_k': 2,
    'aux_loss_coeff': 0.01,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    batch = CONFIG['batch']
    seq_len = CONFIG['seq_len']
    hidden_dim = CONFIG['hidden_dim']
    num_experts = CONFIG['num_experts']
    
    hidden_states = jax.random.normal(k1, (batch, seq_len, hidden_dim), dtype=dtype)
    router_weights = jax.random.normal(k2, (hidden_dim, num_experts), dtype=dtype) * 0.02
    return (hidden_states, router_weights)

def workload(hidden_states, router_weights):
    num_experts = CONFIG['num_experts']
    top_k = CONFIG['top_k']
    aux_loss_coeff = CONFIG['aux_loss_coeff']
    eps = CONFIG['eps']
    
    orig_dtype = hidden_states.dtype
    tokens = hidden_states.reshape(-1, hidden_states.shape[-1])
    num_tokens = tokens.shape[0]
    
    # Router projection accumulated in float32
    logits = jnp.dot(tokens, router_weights, preferred_element_type=jnp.float32)
    
    # Softmax over experts in float32
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Top-2 selection across experts
    top_k_gates, top_k_indices = lax.top_k(probs, top_k)
    
    # Normalize selected gates so they sum to 1 per token
    gate_sum = jnp.sum(top_k_gates, axis=-1, keepdims=True) + eps
    normalized_gates = (top_k_gates / gate_sum).astype(orig_dtype)
    
    # Switch-style auxiliary load-balancing loss in float32
    # P_i: mean router probability allocated to expert i across all tokens
    prob_per_expert = jnp.mean(probs, axis=0)
    
    # f_i: fraction of tokens routed to expert i among the top-k selections
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.float32)
    tokens_per_expert = jnp.sum(expert_mask, axis=(0, 1)) / (num_tokens * top_k)
    
    # Scaled dot product of f_i and P_i
    aux_loss = jnp.sum(prob_per_expert * tokens_per_expert) * num_experts * aux_loss_coeff
    
    out_shape = hidden_states.shape[:-1] + (top_k,)
    return normalized_gates.reshape(out_shape), top_k_indices.reshape(out_shape), aux_loss
