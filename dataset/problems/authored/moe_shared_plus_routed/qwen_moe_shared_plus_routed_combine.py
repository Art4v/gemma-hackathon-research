"""Qwen-MoE layer combine: router softmax top-k gates * precomputed routed expert outputs + sigmoid-gated shared-expert SwiGLU MLP."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'qwen_moe_shared_plus_routed_combine',
    'num_tokens': 4096,
    'hidden_dim': 2048,
    'num_experts': 60,
    'top_k': 4,
    'shared_intermediate': 5632,
    'weight_scale': 0.02,
    'norm_topk_prob': False,
}


def create_inputs(dtype=jnp.bfloat16):
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    E = CONFIG['num_experts']
    K = CONFIG['top_k']
    F = CONFIG['shared_intermediate']
    s = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(keys[0], (T, D), dtype)
    expert_out = jax.random.normal(keys[1], (T, K, D), dtype)
    w_router = jax.random.normal(keys[2], (D, E), dtype) * s
    w_shared_gate = jax.random.normal(keys[3], (D, 1), dtype) * s
    w_gate = jax.random.normal(keys[4], (D, F), dtype) * s
    w_up = jax.random.normal(keys[5], (D, F), dtype) * s
    w_down = jax.random.normal(keys[6], (F, D), dtype) * s
    return x, expert_out, w_router, w_shared_gate, w_gate, w_up, w_down


def workload(x, expert_out, w_router, w_shared_gate, w_gate, w_up, w_down):
    K = expert_out.shape[1]
    f32 = jnp.float32

    # Router gating: fp32 softmax over experts, top-k probabilities as gates.
    logits = jnp.dot(x, w_router, preferred_element_type=f32)            # (T, E)
    probs = jax.nn.softmax(logits, axis=-1)
    gates, _ = lax.top_k(probs, K)                                        # (T, K)
    if CONFIG['norm_topk_prob']:
        gates = gates / jnp.sum(gates, axis=-1, keepdims=True)

    # Weighted combine of precomputed routed expert outputs (slot order = top-k order).
    routed = jnp.einsum('tk,tkd->td', gates, expert_out.astype(f32))      # (T, D)

    # Shared expert SwiGLU MLP.
    g = jnp.dot(x, w_gate, preferred_element_type=f32)
    u = jnp.dot(x, w_up, preferred_element_type=f32)
    h = (jax.nn.silu(g) * u).astype(x.dtype)
    shared = jnp.dot(h, w_down, preferred_element_type=f32)               # (T, D)

    # Per-token sigmoid shared-expert gate.
    sg = jax.nn.sigmoid(jnp.dot(x, w_shared_gate, preferred_element_type=f32))  # (T, 1)

    out = routed + sg * shared
    return out.astype(x.dtype)
