"""DeepSeek-V3 group-limited MoE router with bias-adjusted top-k selection and sigmoid renormalization."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'deepseek_v3_group_limited_router',
    'num_tokens': 16384,
    'hidden_dim': 7168,
    'num_experts': 256,
    'num_groups': 8,
    'topk_groups': 4,
    'topk': 8,
    'topk_group_score': 2,
    'routed_scaling_factor': 2.5,
    'eps': 1e-20,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    num_tokens = CONFIG['num_tokens']
    hidden_dim = CONFIG['hidden_dim']
    num_experts = CONFIG['num_experts']

    x = jax.random.normal(k1, (num_tokens, hidden_dim), dtype=dtype)
    w_router = jax.random.normal(k2, (hidden_dim, num_experts), dtype=dtype) * 0.02
    e_score_correction_bias = jax.random.normal(k3, (num_experts,), dtype=jnp.float32) * 0.01
    return x, w_router, e_score_correction_bias


def workload(x, w_router, e_score_correction_bias):
    num_tokens = x.shape[0]
    num_experts = CONFIG['num_experts']
    num_groups = CONFIG['num_groups']
    topk_groups = CONFIG['topk_groups']
    topk = CONFIG['topk']
    topk_group_score = CONFIG['topk_group_score']
    routed_scaling_factor = CONFIG['routed_scaling_factor']
    eps = CONFIG['eps']
    experts_per_group = num_experts // num_groups

    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)
    scores = jax.nn.sigmoid(logits)

    scores_for_choice = scores + e_score_correction_bias[None, :]
    grouped_scores = scores_for_choice.reshape(num_tokens, num_groups, experts_per_group)

    top2_in_group, _ = lax.top_k(grouped_scores, topk_group_score)
    group_scores = jnp.sum(top2_in_group, axis=-1)

    _, group_idx = lax.top_k(group_scores, topk_groups)

    group_mask = jax.nn.one_hot(group_idx, num_groups, dtype=jnp.float32).sum(axis=1)
    score_mask = jnp.repeat(group_mask, experts_per_group, axis=1)

    masked_scores = jnp.where(score_mask > 0.0, scores_for_choice, -jnp.inf)
    _, topk_idx = lax.top_k(masked_scores, topk)

    topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
    topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + eps)
    topk_weights = (topk_weights * routed_scaling_factor).astype(x.dtype)

    return topk_weights, topk_idx
