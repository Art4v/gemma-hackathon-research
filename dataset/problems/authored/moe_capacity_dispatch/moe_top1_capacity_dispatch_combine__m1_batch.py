"""[m1 M1_batch variant of authored/moe_capacity_dispatch/moe_top1_capacity_dispatch_combine] Top-1 capacity dispatch/combine now runs over a leading axis of G GShard routing groups, each with its own per-group capacity, dispatch buffer [G,E,C,H] and output [G,S,H]."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_top1_capacity_dispatch_combine_m1_batch',
    'num_groups': 4,
    'group_size': 4096,
    'hidden': 2048,
    'num_experts': 64,
    'capacity_factor': 1.25,
}


def _capacity():
    # Capacity is per routing group.
    s = CONFIG['group_size']
    e = CONFIG['num_experts']
    return int(math.ceil(s / e * CONFIG['capacity_factor']))


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kr = jax.random.split(key, 2)
    g, s = CONFIG['num_groups'], CONFIG['group_size']
    h, e = CONFIG['hidden'], CONFIG['num_experts']
    x = jax.random.normal(kx, (g, s, h), dtype)
    w_router = jax.random.normal(kr, (h, e), dtype) * 0.02
    expert_scale = jnp.ones((e, h), dtype)
    return x, w_router, expert_scale


def _route_group(x, w_router, expert_scale):
    t, h = x.shape
    e = w_router.shape[1]
    c = _capacity()

    # Router: float32 accumulation and softmax.
    logits = jnp.dot(x, w_router, preferred_element_type=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    expert_idx = jnp.argmax(probs, axis=-1).astype(jnp.int32)          # [T]
    gate = jnp.max(probs, axis=-1)                                      # [T] f32

    # Slot position within the chosen expert (token order priority).
    onehot = jax.nn.one_hot(expert_idx, e, dtype=jnp.int32)             # [T,E]
    pos_in_expert = jnp.cumsum(onehot, axis=0) * onehot                 # [T,E]
    position = jnp.sum(pos_in_expert, axis=-1) - 1                      # [T]
    keep = position < c                                                 # [T]

    # Dispatch: scatter kept tokens into flat [E*C] slots; overflow -> dump slot.
    flat_idx = jnp.where(keep, expert_idx * c + position, e * c)        # [T]
    buf = jnp.zeros((e * c + 1, h), dtype=x.dtype)
    buf = buf.at[flat_idx].set(x, unique_indices=True, mode='drop')
    dispatch = buf[: e * c].reshape(e, c, h)                            # [E,C,H]

    # Per-expert transform.
    expert_out = (dispatch.astype(jnp.float32)
                  * expert_scale.astype(jnp.float32)[:, None, :])       # [E,C,H] f32

    # Combine: gather each token's slot, weight by gate, zero dropped tokens.
    gather_idx = jnp.where(keep, flat_idx, 0)
    gathered = expert_out.reshape(e * c, h)[gather_idx]                 # [T,H] f32
    weight = jnp.where(keep, gate, 0.0)[:, None]
    out = (gathered * weight).astype(x.dtype)
    return dispatch, out


def workload(x, w_router, expert_scale):
    # Router weights and expert scales are shared by all groups.
    return jax.vmap(_route_group, in_axes=(0, None, None))(x, w_router, expert_scale)
