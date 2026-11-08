"""Qwen-MoE layer combine: sigmoid-gated shared-expert SwiGLU MLP + gate-weighted sum of top-k routed expert outputs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'qwen_moe_shared_routed_combine',
    'num_tokens': 4096,
    'hidden_size': 2048,
    'shared_intermediate_size': 5632,
    'num_experts': 60,
    'top_k': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_size']
    F = CONFIG['shared_intermediate_size']
    E = CONFIG['num_experts']
    K = CONFIG['top_k']
    s = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(keys[0], (T, D), dtype)
    expert_out = jax.random.normal(keys[1], (T, K, D), dtype)
    # Router: softmax over all experts, take top-k probs (no renormalization, as in Qwen1.5-MoE).
    router_logits = jax.random.normal(keys[2], (T, E), jnp.float32)
    probs = jax.nn.softmax(router_logits, axis=-1)
    gates, _ = jax.lax.top_k(probs, K)
    gates = gates.astype(dtype)
    w_gate = jax.random.normal(keys[3], (D, F), dtype) * s
    w_up = jax.random.normal(keys[4], (D, F), dtype) * s
    w_down = jax.random.normal(keys[5], (F, D), dtype) * s
    w_shared_gate = jax.random.normal(keys[6], (D, 1), dtype) * s
    return x, expert_out, gates, w_gate, w_up, w_down, w_shared_gate


def workload(x, expert_out, gates, w_gate, w_up, w_down, w_shared_gate):
    f32 = jnp.float32
    # Shared expert SwiGLU MLP (fp32 accumulation).
    h_gate = jnp.dot(x, w_gate, preferred_element_type=f32)
    h_up = jnp.dot(x, w_up, preferred_element_type=f32)
    act = (jax.nn.silu(h_gate) * h_up).astype(x.dtype)
    shared = jnp.dot(act, w_down, preferred_element_type=f32)
    # Per-token sigmoid shared-expert gate.
    sg = jax.nn.sigmoid(jnp.dot(x, w_shared_gate, preferred_element_type=f32))
    # Gate-weighted combine of top-k routed expert outputs.
    routed = jnp.einsum('tk,tkd->td', gates.astype(f32), expert_out.astype(f32))
    out = routed + sg * shared
    return out.astype(x.dtype)
