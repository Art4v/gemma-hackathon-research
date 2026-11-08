"""DeepSeek-V3 MoE dense shared-expert SwiGLU MLP fused with top-k routed expert weighted combine and residual add."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'deepseek_moe_shared_expert_routed_combine',
    'num_tokens': 4096,
    'hidden_dim': 4096,
    'shared_inter_dim': 2048,
    'top_k': 6,
    'routed_scaling_factor': 2.5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    T = CONFIG['num_tokens']
    H = CONFIG['hidden_dim']
    I = CONFIG['shared_inter_dim']
    K = CONFIG['top_k']

    hidden_states = jax.random.normal(k1, (T, H), dtype=dtype)
    residual = jax.random.normal(k2, (T, H), dtype=dtype)
    routed_expert_outs = jax.random.normal(k3, (T, K, H), dtype=dtype)
    raw_gates = jax.random.uniform(k4, (T, K), dtype=jnp.float32)
    routed_gates = (raw_gates / jnp.sum(raw_gates, axis=-1, keepdims=True)).astype(dtype)

    w_gate = jax.random.normal(k5, (H, I), dtype=dtype) * 0.02
    w_up = jax.random.normal(k6, (H, I), dtype=dtype) * 0.02
    w_down = jax.random.normal(k7, (I, H), dtype=dtype) * 0.02

    return hidden_states, residual, routed_expert_outs, routed_gates, w_gate, w_up, w_down

def workload(hidden_states, residual, routed_expert_outs, routed_gates, w_gate, w_up, w_down):
    out_dtype = hidden_states.dtype
    scale = CONFIG['routed_scaling_factor']

    gate_proj = jnp.dot(hidden_states, w_gate, preferred_element_type=jnp.float32)
    up_proj = jnp.dot(hidden_states, w_up, preferred_element_type=jnp.float32)
    shared_inter = (jax.nn.silu(gate_proj) * up_proj).astype(out_dtype)
    shared_out = jnp.dot(shared_inter, w_down, preferred_element_type=jnp.float32)

    gates_f32 = routed_gates.astype(jnp.float32)[:, :, None]
    routed_f32 = routed_expert_outs.astype(jnp.float32)
    routed_combined = jnp.sum(routed_f32 * gates_f32, axis=1) * scale

    total_out = residual.astype(jnp.float32) + shared_out + routed_combined
    return total_out.astype(out_dtype)
