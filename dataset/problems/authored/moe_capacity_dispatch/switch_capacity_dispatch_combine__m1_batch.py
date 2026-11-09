"""[m1 M1_batch variant of authored/moe_capacity_dispatch/switch_capacity_dispatch_combine] Switch top-1 capacity dispatch -> expert GELU FFN -> gated combine now runs over a leading axis of G routing groups, each with its own per-group capacity and dispatch buffer [G,E,C,H]."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'switch_capacity_dispatch_combine_m1_batch',
    'num_groups': 4,
    'group_size': 4096,
    'hidden': 2048,
    'num_experts': 16,
    'ffn': 2048,
    'capacity_factor': 1.25,
    'capacity': int(math.ceil(1.25 * 4096 / 16)),  # 320 per group
}


def create_inputs(dtype=jnp.bfloat16):
    G, S = CONFIG['num_groups'], CONFIG['group_size']
    H, E, F = CONFIG['hidden'], CONFIG['num_experts'], CONFIG['ffn']
    key = jax.random.key(0)
    kx, kr, k1, k2 = jax.random.split(key, 4)
    x = jax.random.normal(kx, (G, S, H), dtype)
    w_router = jax.random.normal(kr, (H, E), dtype) * 0.02
    w1 = jax.random.normal(k1, (E, H, F), dtype) * 0.02
    w2 = jax.random.normal(k2, (E, F, H), dtype) * 0.02
    return x, w_router, w1, w2


def _moe_group(x, w_router, w1, w2):
    T, H = x.shape
    E = w_router.shape[1]
    C = CONFIG['capacity']

    # Router (float32).
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)  # [T, E]
    probs = jax.nn.softmax(logits, axis=-1)
    expert_idx = jnp.argmax(probs, axis=-1).astype(jnp.int32)  # [T]
    gate = jnp.take_along_axis(probs, expert_idx[:, None], axis=-1)[:, 0]  # [T] f32

    # Capacity slot assignment: position of each token within its expert (token order).
    onehot = jax.nn.one_hot(expert_idx, E, dtype=jnp.int32)  # [T, E]
    pos_all = jnp.cumsum(onehot, axis=0) - 1  # [T, E]
    slot = jnp.take_along_axis(pos_all, expert_idx[:, None], axis=-1)[:, 0]  # [T]
    keep = slot < C

    # Dispatch: scatter kept tokens into [E*C (+1 dummy), H].
    dummy = E * C
    flat_idx = jnp.where(keep, expert_idx * C + slot, dummy)
    buf = jnp.zeros((E * C + 1, H), dtype=x.dtype).at[flat_idx].set(x)
    dispatched = buf[:dummy].reshape(E, C, H)

    # Per-expert GELU FFN (bf16 operands, f32 accumulation).
    h = jnp.einsum('ech,ehf->ecf', dispatched, w1, preferred_element_type=jnp.float32)
    h = jax.nn.gelu(h, approximate=True).astype(x.dtype)
    eo = jnp.einsum('ecf,efh->ech', h, w2, preferred_element_type=jnp.float32)  # [E, C, H] f32

    # Combine: gather each token's expert output, weight by gate; dropped tokens -> 0.
    eo_flat = jnp.concatenate([eo.reshape(E * C, H), jnp.zeros((1, H), jnp.float32)], axis=0)
    gathered = eo_flat[flat_idx]  # [T, H] f32
    out = gathered * jnp.where(keep, gate, 0.0)[:, None]
    return out.astype(x.dtype), dispatched


def workload(x, w_router, w1, w2):
    # Router and expert weights are shared by all groups.
    return jax.vmap(_moe_group, in_axes=(0, None, None, None))(x, w_router, w1, w2)
