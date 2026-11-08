"""MoE Top-K router: linear gating projection, softmax, top-k selection, and weight renormalization."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_topk_router',
    'num_tokens': 32768,
    'hidden_dim': 4096,
    'num_experts': 160,
    'top_k': 8,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    x = jax.random.normal(k1, (CONFIG['num_tokens'], CONFIG['hidden_dim']), dtype=dtype)
    w_router = jax.random.normal(k2, (CONFIG['hidden_dim'], CONFIG['num_experts']), dtype=dtype) * 0.02
    return (x, w_router)

def workload(x, w_router):
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    topk_weights, topk_indices = lax.top_k(probs, CONFIG['top_k'])
    norm_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + CONFIG['eps'])
    return norm_weights.astype(x.dtype), topk_indices
