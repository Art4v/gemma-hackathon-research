"""DeepSeek-V3 MoE router: x @ W_router -> sigmoid -> +bias -> group-limited top-k (top-2 sum per group, top-4 groups) -> top-8 experts -> renormalize unbiased weights * routed_scaling_factor."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_sigmoid_grouped_topk_router',
    'num_tokens': 16384,
    'hidden_size': 7168,
    'num_experts': 256,
    'n_group': 8,
    'topk_group': 4,
    'top_k': 8,
    'group_score_topk': 2,
    'routed_scaling_factor': 2.5,
    'norm_eps': 1e-20,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    T = CONFIG['num_tokens']
    H = CONFIG['hidden_size']
    E = CONFIG['num_experts']
    x = jax.random.normal(kx, (T, H), dtype=dtype)
    w_router = jax.random.normal(kw, (H, E), dtype=dtype) * 0.02
    e_score_correction_bias = jax.random.normal(kb, (E,), dtype=jnp.float32) * 0.02
    return x, w_router, e_score_correction_bias


def workload(x, w_router, e_score_correction_bias):
    T = x.shape[0]
    E = w_router.shape[1]
    G = CONFIG['n_group']
    topk_group = CONFIG['topk_group']
    top_k = CONFIG['top_k']
    gk = CONFIG['group_score_topk']
    experts_per_group = E // G

    # Router GEMM with float32 accumulation.
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)  # [T, E]
    scores = jax.nn.sigmoid(logits)  # unbiased affinities, f32

    # Bias only influences selection.
    scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)[None, :]

    # Group scores: sum of top-gk experts within each group.
    grouped = scores_for_choice.reshape(T, G, experts_per_group)
    group_top_vals, _ = lax.top_k(grouped, gk)  # [T, G, gk]
    group_scores = jnp.sum(group_top_vals, axis=-1)  # [T, G]

    # Keep top `topk_group` groups.
    _, group_idx = lax.top_k(group_scores, topk_group)  # [T, topk_group]
    group_mask = jnp.sum(jax.nn.one_hot(group_idx, G, dtype=jnp.float32), axis=1) > 0  # [T, G]
    expert_mask = jnp.repeat(group_mask, experts_per_group, axis=1)  # [T, E]

    masked_scores = jnp.where(expert_mask, scores_for_choice, -jnp.inf)

    # Global top-k among eligible experts.
    _, topk_indices = lax.top_k(masked_scores, top_k)  # [T, top_k]
    topk_weights = jnp.take_along_axis(scores, topk_indices, axis=1)  # unbiased

    denom = jnp.sum(topk_weights, axis=-1, keepdims=True) + CONFIG['norm_eps']
    topk_weights = topk_weights / denom * CONFIG['routed_scaling_factor']

    return topk_weights.astype(jnp.float32), topk_indices.astype(jnp.int32)
