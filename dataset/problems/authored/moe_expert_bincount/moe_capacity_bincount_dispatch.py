"""MoE routing: softmax -> top-k -> expert bincount + group offsets -> capacity-limited rank -> gate-scaled dense dispatch mask."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_capacity_bincount_dispatch',
    'num_tokens': 131072,
    'num_experts': 256,
    'top_k': 8,
    'capacity_factor': 1.25,
    'capacity': 5120,  # ceil(1.25 * 131072 * 8 / 256)
    'logit_scale': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_logits, = jax.random.split(key, 1)
    router_logits = jax.random.normal(
        k_logits, (CONFIG['num_tokens'], CONFIG['num_experts']), dtype
    ) * CONFIG['logit_scale']
    return (router_logits.astype(dtype),)


def workload(router_logits):
    T, E = router_logits.shape
    K = CONFIG['top_k']
    C = CONFIG['capacity']

    # Router probabilities and top-k gates (float32 accumulation).
    probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    gates, idx = lax.top_k(probs, K)                      # (T, K)
    gates = gates / jnp.sum(gates, axis=-1, keepdims=True)
    idx = idx.astype(jnp.int32)

    # k-major priority order: all 1st choices, then all 2nd choices, ...
    flat_e = idx.T.reshape(-1)                            # (K*T,)
    N = flat_e.shape[0]

    # Per-expert bincount and exclusive group offsets.
    counts = jnp.bincount(flat_e, length=E).astype(jnp.int32)
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)]
    )                                                     # (E+1,)

    # Rank of each choice within its expert (stable => priority preserved).
    order = jnp.argsort(flat_e, stable=True).astype(jnp.int32)
    sorted_e = flat_e[order]
    ranks_sorted = jnp.arange(N, dtype=jnp.int32) - offsets[sorted_e]
    ranks_flat = jnp.zeros((N,), jnp.int32).at[order].set(ranks_sorted)
    rank = ranks_flat.reshape(K, T).T                     # (T, K)

    # Capacity truncation.
    keep = rank < C
    slot = jnp.where(keep, rank, jnp.int32(-1))
    kept_load = jnp.minimum(counts, jnp.int32(C))

    # Dense gate-scaled dispatch/combine mask (T, E).
    w = jnp.where(keep, gates, jnp.float32(0.0))
    rows = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[:, None], (T, K))
    dispatch = jnp.zeros((T, E), jnp.float32).at[rows, idx].add(w)

    return counts, offsets, kept_load, slot, dispatch
