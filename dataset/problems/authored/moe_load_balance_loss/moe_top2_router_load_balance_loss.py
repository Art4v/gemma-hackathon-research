"""MoE router: linear logits -> float32 softmax -> top-2 selection with renormalized gates -> Switch-style auxiliary load-balancing loss."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_top2_router_load_balance_loss',
    'batch': 8,
    'seq_len': 4096,
    'd_model': 4096,
    'num_experts': 64,
    'top_k': 2,
    'aux_loss_coef': 0.01,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    tokens = CONFIG['batch'] * CONFIG['seq_len']
    x = jax.random.normal(k_x, (tokens, CONFIG['d_model']), dtype=dtype)
    w_router = jax.random.normal(
        k_w, (CONFIG['d_model'], CONFIG['num_experts']), dtype=dtype
    ) * CONFIG['weight_scale']
    return x, w_router


def workload(x, w_router):
    num_tokens = x.shape[0]
    num_experts = w_router.shape[1]
    top_k = CONFIG['top_k']

    # Router logits, accumulated in float32.
    logits = lax.dot_general(
        x, w_router,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # (T, E)

    probs = jax.nn.softmax(logits, axis=-1)  # (T, E) float32

    # Top-2 selection, then renormalize the selected probabilities.
    top_probs, top_idx = lax.top_k(probs, top_k)  # (T, k)
    gates = top_probs / jnp.sum(top_probs, axis=-1, keepdims=True)

    # Switch-style load-balancing loss.
    # f_e: fraction of the T*k routing slots assigned to expert e.
    dispatch = jax.nn.one_hot(top_idx, num_experts, dtype=jnp.float32)  # (T, k, E)
    tokens_per_expert = jnp.sum(dispatch, axis=(0, 1))  # (E,)
    f = tokens_per_expert / (num_tokens * top_k)
    # P_e: mean router probability for expert e.
    p = jnp.mean(probs, axis=0)  # (E,)
    aux_loss = CONFIG['aux_loss_coef'] * num_experts * jnp.sum(f * p)

    return gates.astype(x.dtype), top_idx.astype(jnp.int32), aux_loss.astype(jnp.float32)
