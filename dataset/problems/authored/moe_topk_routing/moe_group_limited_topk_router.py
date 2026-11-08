"""DeepSeek-V2/V3 group-limited MoE Top-K router: router GEMM, softmax, group top-k masking, expert top-k selection, and weight renormalization."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_group_limited_topk_router',
    'num_tokens': 16384,
    'hidden_dim': 7168,
    'num_experts': 256,
    'num_groups': 8,
    'topk_groups': 4,
    'topk_experts_per_group_score': 2,
    'topk': 8,
    'routed_scaling_factor': 2.5,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    E = CONFIG['num_experts']
    x = jax.random.normal(k1, (T, D), dtype=dtype)
    w_router = jax.random.normal(k2, (D, E), dtype=dtype) * 0.02
    e_score_correction_bias = jax.random.normal(k3, (E,), dtype=jnp.float32) * 0.01
    return x, w_router, e_score_correction_bias

def workload(x, w_router, e_score_correction_bias):
    T = x.shape[0]
    E = CONFIG['num_experts']
    G = CONFIG['num_groups']
    E_per_G = E // G
    topk_g = CONFIG['topk_groups']
    topk_sub = CONFIG['topk_experts_per_group_score']
    topk = CONFIG['topk']
    scale = CONFIG['routed_scaling_factor']
    eps = CONFIG['eps']

    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)

    scores_for_choice = probs + e_score_correction_bias[None, :]
    scores_grouped = jnp.reshape(scores_for_choice, (T, G, E_per_G))

    top_sub_vals, _ = lax.top_k(scores_grouped, topk_sub)
    group_scores = jnp.sum(top_sub_vals, axis=-1)

    _, top_group_idx = lax.top_k(group_scores, topk_g)
    group_mask = jnp.sum(jax.nn.one_hot(top_group_idx, G, dtype=jnp.bool_), axis=-2)

    masked_scores = jnp.where(group_mask[:, :, None], scores_grouped, -jnp.inf)
    masked_scores_flat = jnp.reshape(masked_scores, (T, E))

    _, topk_indices = lax.top_k(masked_scores_flat, topk)
    topk_weights = jnp.take_along_axis(probs, topk_indices, axis=-1)

    topk_weights = (topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + eps)) * scale
    return topk_weights.astype(x.dtype), topk_indices
