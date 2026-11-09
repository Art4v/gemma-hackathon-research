"""[m1 M1_batch variant of authored/moe_load_balance_loss/top2_router_load_balance_loss] Router -> softmax -> renormalized top-2 gates, now over a leading batch axis of sequences with a sequence-wise Switch/GShard load-balancing loss per sequence."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'top2_router_load_balance_loss_m1_batch',
    'batch': 4,
    'seq_len': 4096,
    'hidden_dim': 4096,
    'num_experts': 64,
    'top_k': 2,
    'aux_loss_coef': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.normal(
        k_x, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_dim']), dtype)
    w_router = jax.random.normal(
        k_w, (CONFIG['hidden_dim'], CONFIG['num_experts']), dtype) * 0.02
    return x, w_router


def _route_sequence(x, w_router):
    num_experts = w_router.shape[1]
    top_k = CONFIG['top_k']
    alpha = CONFIG['aux_loss_coef']

    # Router logits with float32 accumulation.
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)  # [S, E]
    probs = jax.nn.softmax(logits, axis=-1)                              # [S, E] f32

    # Top-k selection and renormalization of selected gates.
    top_vals, top_idx = lax.top_k(probs, top_k)                          # [S, K]
    gates = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)

    # Switch/GShard load-balancing loss over this sequence's tokens.
    dispatch = jnp.sum(
        jax.nn.one_hot(top_idx, num_experts, dtype=jnp.float32), axis=1)  # [S, E]
    tokens_per_expert = jnp.mean(dispatch, axis=0) / top_k               # f_e, sums to 1
    prob_per_expert = jnp.mean(probs, axis=0)                            # P_e
    aux_loss = alpha * num_experts * jnp.sum(tokens_per_expert * prob_per_expert)

    return gates.astype(x.dtype), top_idx.astype(jnp.int32), aux_loss.astype(jnp.float32)


def workload(x, w_router):
    # The router weight is shared; the balance loss is per sequence.
    return jax.vmap(_route_sequence, in_axes=(0, None))(x, w_router)
