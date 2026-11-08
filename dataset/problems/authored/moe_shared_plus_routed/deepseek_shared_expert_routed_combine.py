"""DeepSeek-V3 MoE shared-expert SwiGLU MLP fused with top-k routed expert gated combine and residual add."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'deepseek_shared_expert_routed_combine',
    'num_tokens': 4096,
    'hidden_size': 4096,
    'shared_intermediate_size': 2048,
    'top_k': 8,
    'routed_scaling_factor': 2.5,
    'norm_eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_routed, k_gates, k_wg, k_wu, k_wd = jax.random.split(key, 6)
    T = CONFIG['num_tokens']
    H = CONFIG['hidden_size']
    I = CONFIG['shared_intermediate_size']
    K = CONFIG['top_k']

    hidden_states = jax.random.normal(k_x, (T, H), dtype=dtype)
    routed_outputs = jax.random.normal(k_routed, (T, K, H), dtype=dtype)
    gate_scores = jax.random.uniform(
        k_gates, (T, K), dtype=jnp.float32, minval=0.01, maxval=1.0
    ).astype(dtype)

    w_gate = jax.random.normal(k_wg, (H, I), dtype=dtype) * 0.02
    w_up = jax.random.normal(k_wu, (H, I), dtype=dtype) * 0.02
    w_down = jax.random.normal(k_wd, (I, H), dtype=dtype) * 0.02
    rms_weight = jnp.ones((H,), dtype=dtype)

    return (
        hidden_states,
        routed_outputs,
        gate_scores,
        w_gate,
        w_up,
        w_down,
        rms_weight,
    )


def workload(
    hidden_states,
    routed_outputs,
    gate_scores,
    w_gate,
    w_up,
    w_down,
    rms_weight,
):
    orig_dtype = hidden_states.dtype
    eps = CONFIG['norm_eps']
    scale = CONFIG['routed_scaling_factor']

    x_f32 = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed_x = x_f32 * jax.lax.rsqrt(variance + eps)
    normed_x = (normed_x * rms_weight.astype(jnp.float32)).astype(orig_dtype)

    gate_proj = jnp.dot(
        normed_x, w_gate, preferred_element_type=jnp.float32
    )
    up_proj = jnp.dot(
        normed_x, w_up, preferred_element_type=jnp.float32
    )
    shared_act = (jax.nn.silu(gate_proj) * up_proj).astype(orig_dtype)
    shared_out = jnp.dot(
        shared_act, w_down, preferred_element_type=jnp.float32
    )

    scores_f32 = gate_scores.astype(jnp.float32)
    scores_sum = jnp.sum(scores_f32, axis=-1, keepdims=True) + eps
    norm_gates = (scores_f32 / scores_sum) * scale

    routed_f32 = routed_outputs.astype(jnp.float32)
    routed_combined = jnp.sum(routed_f32 * norm_gates[:, :, None], axis=1)

    total_out = x_f32 + shared_out + routed_combined
    return total_out.astype(orig_dtype)
