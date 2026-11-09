"""[m1 M1_batch variant of authored/moe_expert_bincount/moe_capacity_dispatch_mask] Per-group bincount -> offsets -> slot positions -> capacity drop -> dense combine tensor, now over a leading batch axis of B sequences, each split into its own 512-token routing groups."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_capacity_dispatch_mask_m1_batch',
    'batch': 4,
    'seq_len': 12288,
    'num_experts': 384,
    'top_k': 8,
    'group_size': 512,
    'capacity_factor': 1.25,
    'capacity': int(math.ceil(512 * 8 / 384 * 1.25)),  # 14
    'renorm_eps': 1e-9,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    T = CONFIG['seq_len']
    E = CONFIG['num_experts']
    k = CONFIG['top_k']
    key = jax.random.key(0)
    k_logits, _ = jax.random.split(key)
    logits = jax.random.normal(k_logits, (B, T, E), jnp.float32) * 2.0
    # The router runs in float32 no matter what dtype is (keeps this input above 50 MB).
    router_probs = jax.nn.softmax(logits, axis=-1).astype(jnp.float32)
    # Top-k indices so each token picks k different experts.
    _, expert_indices = lax.top_k(logits, k)
    expert_indices = expert_indices.astype(jnp.int32)
    return router_probs, expert_indices


def _route_sequence(router_probs, expert_indices):
    T, E = router_probs.shape
    k = expert_indices.shape[1]
    S = CONFIG['group_size']
    C = CONFIG['capacity']
    G = T // S
    out_dtype = jnp.bfloat16

    # Gates: gather the top-k probabilities and renormalise them in float32.
    gates = jnp.take_along_axis(router_probs.astype(jnp.float32), expert_indices, axis=-1)
    gates = gates / (jnp.sum(gates, axis=-1, keepdims=True) + CONFIG['renorm_eps'])

    idx = expert_indices.reshape(G, S, k)
    gates = gates.reshape(G, S, k)

    # Priority order: all top-1 choices first, then top-2, and so on (GShard).
    idx_ks = jnp.transpose(idx, (0, 2, 1)).reshape(G, k * S)            # (G, kS)
    onehot = jax.nn.one_hot(idx_ks, E, dtype=jnp.int32)                  # (G, kS, E)

    # Tokens per expert in each group (bincount) and exclusive offsets.
    expert_counts = jnp.sum(onehot, axis=1)                              # (G, E)
    group_offsets = jnp.cumsum(expert_counts, axis=-1) - expert_counts   # (G, E)

    # Position of each choice inside its expert's buffer.
    running = jnp.cumsum(onehot, axis=1) - 1                             # (G, kS, E)
    pos_ks = jnp.sum(running * onehot, axis=-1)                          # (G, kS)
    pos = jnp.transpose(pos_ks.reshape(G, k, S), (0, 2, 1))              # (G, S, k)

    # Capacity limit.
    keep = pos < C
    scaled = jnp.where(keep, gates, 0.0)                                 # f32 (G, S, k)
    safe_pos = jnp.where(keep, pos, 0)

    # Dense combine tensor; dispatch mask = combine > 0.
    # Each kept (g, e, pos) slot is written by exactly one choice, so the scatter is collision-free.
    g_ix = jnp.broadcast_to(jnp.arange(G, dtype=jnp.int32)[:, None, None], (G, S, k))
    s_ix = jnp.broadcast_to(jnp.arange(S, dtype=jnp.int32)[None, :, None], (G, S, k))
    combine = jnp.zeros((G, S, E, C), jnp.float32)
    combine = combine.at[g_ix, s_ix, idx, safe_pos].add(scaled)
    combine = combine.astype(out_dtype)

    return combine, expert_counts, group_offsets


def workload(router_probs, expert_indices):
    return jax.vmap(_route_sequence)(router_probs, expert_indices)
