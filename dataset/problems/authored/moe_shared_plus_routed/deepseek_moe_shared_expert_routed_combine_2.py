"""DeepSeek-V3 style MoE output combine: dense shared-expert SwiGLU MLP fused with top-k routed expert unpermute, gate-weighted reduction, and residual add."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'deepseek_moe_shared_expert_routed_combine',
    'num_tokens': 8192,
    'hidden_size': 4096,
    'shared_intermediate_size': 2048,
    'top_k': 8,
    'routed_scaling_factor': 2.5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_routed, k_gates, k_perm, k_wg, k_wu, k_wd = jax.random.split(key, 7)

    T = CONFIG['num_tokens']
    H = CONFIG['hidden_size']
    I = CONFIG['shared_intermediate_size']
    K = CONFIG['top_k']

    hidden_states = jax.random.normal(k_x, (T, H), dtype=dtype)
    routed_expert_outputs = jax.random.normal(k_routed, (T * K, H), dtype=dtype)

    raw_gates = jax.random.uniform(k_gates, (T, K), dtype=jnp.float32, minval=0.01, maxval=1.0)
    routing_weights = (raw_gates / jnp.sum(raw_gates, axis=-1, keepdims=True)).astype(dtype)

    unpermute_indices = jax.random.permutation(k_perm, jnp.arange(T * K, dtype=jnp.int32)).reshape(T, K)

    w_gate = jax.random.normal(k_wg, (H, I), dtype=dtype) * 0.02
    w_up = jax.random.normal(k_wu, (H, I), dtype=dtype) * 0.02
    w_down = jax.random.normal(k_wd, (I, H), dtype=dtype) * 0.02

    return (
        hidden_states,
        routed_expert_outputs,
        routing_weights,
        unpermute_indices,
        w_gate,
        w_up,
        w_down,
    )


def workload(
    hidden_states,
    routed_expert_outputs,
    routing_weights,
    unpermute_indices,
    w_gate,
    w_up,
    w_down,
):
    orig_dtype = hidden_states.dtype
    scale = CONFIG['routed_scaling_factor']

    gate = jnp.dot(hidden_states, w_gate, preferred_element_type=jnp.float32)
    up = jnp.dot(hidden_states, w_up, preferred_element_type=jnp.float32)
    shared_hidden = (jax.nn.silu(gate) * up).astype(orig_dtype)
    shared_out = jnp.dot(shared_hidden, w_down, preferred_element_type=jnp.float32)

    gathered_routed = jnp.take(routed_expert_outputs, unpermute_indices, axis=0).astype(jnp.float32)
    weights = routing_weights.astype(jnp.float32)[:, :, None] * scale
    routed_combined = jnp.sum(gathered_routed * weights, axis=1)

    output = hidden_states.astype(jnp.float32) + shared_out + routed_combined
    return output.astype(orig_dtype)
