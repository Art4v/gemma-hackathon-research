"""DeepSeek-V3 MoE router: x @ W_router -> sigmoid -> +bias -> group-limited top-k -> renormalized scaled weights."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_sigmoid_grouped_topk_router',
    'num_tokens': 16384,
    'hidden': 7168,
    'num_experts': 256,
    'n_group': 8,
    'topk_group': 4,
    'group_score_topk': 2,
    'top_k': 8,
    'routed_scaling_factor': 2.5,
    'norm_eps': 1e-20,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    T, D, E = CONFIG['num_tokens'], CONFIG['hidden'], CONFIG['num_experts']
    x = jax.random.normal(k1, (T, D), dtype)
    w_router = jax.random.normal(k2, (D, E), dtype) * 0.02
    # e_score_correction_bias: kept in float32, as in the reference model
    bias = jax.random.normal(k3, (E,), jnp.float32) * 0.02
    return x, w_router, bias


def workload(x, w_router, bias):
    T = x.shape[0]
    E = w_router.shape[1]
    G = CONFIG['n_group']
    k = CONFIG['top_k']

    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)  # (T, E) f32
    scores = jax.nn.sigmoid(logits)
    biased = scores + bias.astype(jnp.float32)[None, :]

    # Group scores: sum of top-`group_score_topk` biased scores within each group.
    grouped = biased.reshape(T, G, E // G)
    top_in_group, _ = lax.top_k(grouped, CONFIG['group_score_topk'])
    group_scores = jnp.sum(top_in_group, axis=-1)  # (T, G)

    # Keep the best `topk_group` groups.
    _, group_idx = lax.top_k(group_scores, CONFIG['topk_group'])  # (T, topk_group)
    group_mask = jnp.sum(jax.nn.one_hot(group_idx, G, dtype=jnp.float32), axis=1) > 0  # (T, G)
    expert_mask = jnp.repeat(group_mask, E // G, axis=1)  # (T, E)

    masked = jnp.where(expert_mask, biased, -jnp.inf)
    _, topk_idx = lax.top_k(masked, k)  # (T, k)

    # Weights from unbiased scores, renormalized and scaled.
    w = jnp.take_along_axis(scores, topk_idx, axis=1)
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + CONFIG['norm_eps'])
    w = w * CONFIG['routed_scaling_factor']
    return w.astype(x.dtype), topk_idx.astype(jnp.int32)
