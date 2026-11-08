"""Qwen-MoE layer combine: router softmax+top-k gates, sigmoid-gated shared-expert SwiGLU MLP, weighted sum of precomputed routed expert outputs."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'qwen_moe_shared_routed_combine',
    'num_tokens': 4096,
    'hidden_size': 2048,
    'num_experts': 60,
    'top_k': 4,
    'shared_intermediate_size': 5632,
    'norm_topk_prob': False,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_size']
    E = CONFIG['num_experts']
    K = CONFIG['top_k']
    F = CONFIG['shared_intermediate_size']
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

    # Router: softmax over all experts in f32, then top-k gate weights.
    logits = jnp.dot(x, w_router, preferred_element_type=f32)
    probs = jax.nn.softmax(logits, axis=-1)
    topk_w, _ = lax.top_k(probs, K)
    if CONFIG['norm_topk_prob']:
        topk_w = topk_w / jnp.sum(topk_w, axis=-1, keepdims=True)

    # Routed combine: weighted sum over the K precomputed expert outputs.
    routed = jnp.einsum('tk,tkd->td', topk_w, expert_out.astype(f32))

    # Shared expert SwiGLU MLP with f32 accumulation.
    g = jnp.dot(x, w_gate, preferred_element_type=f32)
    u = jnp.dot(x, w_up, preferred_element_type=f32)
    h = (jax.nn.silu(g) * u).astype(x.dtype)
    shared = jnp.dot(h, w_down, preferred_element_type=f32)

    # Qwen-style scalar sigmoid gate on the shared expert.
    sg = jax.nn.sigmoid(jnp.dot(x, w_shared_gate, preferred_element_type=f32))

    out = routed + sg * shared
    return out.astype(x.dtype)
