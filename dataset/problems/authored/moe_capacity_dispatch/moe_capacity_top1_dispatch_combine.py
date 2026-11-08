"""Switch-style top-1 MoE routing: router softmax -> top-1 -> capacity slot (cumsum) -> drop overflow -> scatter dispatch [E,C,H] -> gated gather combine."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_capacity_top1_dispatch_combine',
    'num_tokens': 16384,
    'hidden': 4096,
    'num_experts': 64,
    'capacity_factor': 1.25,
    'capacity': int(math.ceil(1.25 * 16384 / 64)),  # 320
}


def create_inputs(dtype=jnp.bfloat16):
    T = CONFIG['num_tokens']
    H = CONFIG['hidden']
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    k_x, k_r, k_e = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k_x, (T, H), dtype)
    router_w = jax.random.normal(k_r, (H, E), dtype) * 0.02
    expert_out = jax.random.normal(k_e, (E, C, H), dtype)
    return x, router_w, expert_out


def workload(x, router_w, expert_out):
    T, H = x.shape
    E = router_w.shape[1]
    C = CONFIG['capacity']

    # Router: fp32 logits and softmax.
    logits = jnp.dot(x, router_w, preferred_element_type=jnp.float32)  # [T, E]
    probs = jax.nn.softmax(logits, axis=-1)
    expert_idx = jnp.argmax(probs, axis=-1).astype(jnp.int32)  # [T]
    gate = jnp.take_along_axis(probs, expert_idx[:, None], axis=-1)[:, 0]  # [T] f32

    # Slot within expert queue: exclusive running count in token order.
    onehot = jax.nn.one_hot(expert_idx, E, dtype=jnp.int32)  # [T, E]
    slots_all = jnp.cumsum(onehot, axis=0) - onehot  # exclusive cumsum
    slot = jnp.take_along_axis(slots_all, expert_idx[:, None], axis=-1)[:, 0]  # [T]
    keep = slot < C  # drop overflow tokens

    # Flat destination index into [E*C]; dropped tokens point out of bounds.
    flat_idx = jnp.where(keep, expert_idx * C + slot, E * C)

    # Dispatch: scatter kept tokens into the zero-padded capacity buffer.
    dispatch = jnp.zeros((E * C, H), dtype=x.dtype)
    dispatch = dispatch.at[flat_idx].set(x, mode='drop', unique_indices=True)
    dispatch = dispatch.reshape(E, C, H)

    # Combine: gather each token's expert output row, scale by gate, zero if dropped.
    eo_flat = expert_out.reshape(E * C, H)
    safe_idx = jnp.minimum(flat_idx, E * C - 1)
    gathered = jnp.take(eo_flat, safe_idx, axis=0).astype(jnp.float32)  # [T, H]
    weight = jnp.where(keep, gate, 0.0)[:, None]
    combined = (gathered * weight).astype(x.dtype)

    return dispatch, combined
