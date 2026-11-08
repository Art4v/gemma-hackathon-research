"""MoE router: x @ W_router -> fp32 softmax -> top-k expert selection -> renormalized gate weights."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_router_softmax_topk_renorm',
    'batch': 8,
    'seq_len': 4096,
    'hidden': 4096,
    'num_experts': 128,
    'top_k': 8,
    'router_init_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key, 2)
    tokens = CONFIG['batch'] * CONFIG['seq_len']
    x = jax.random.normal(k_x, (tokens, CONFIG['hidden']), dtype)
    w_router = (
        jax.random.normal(k_w, (CONFIG['hidden'], CONFIG['num_experts']), dtype)
        * CONFIG['router_init_scale']
    )
    return (x, w_router)


def workload(x, w_router):
    top_k = CONFIG['top_k']
    # Router GEMM with fp32 accumulation.
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)
    # Softmax over experts in fp32.
    probs = jax.nn.softmax(logits, axis=-1)
    # Top-k experts per token (ties -> lower index).
    topk_probs, topk_idx = lax.top_k(probs, top_k)
    # Renormalize selected gate weights to sum to 1.
    denom = jnp.sum(topk_probs, axis=-1, keepdims=True)
    topk_weights = topk_probs / denom
    return topk_idx.astype(jnp.int32), topk_weights.astype(x.dtype)
