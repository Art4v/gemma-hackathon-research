"""Switch Transformer top-1 capacity-limited MoE token dispatch and combine with overflow token dropping."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'switch_capacity_dispatch_combine',
    'num_tokens': 32768,
    'num_experts': 32,
    'capacity': 1280,
    'd_model': 2048,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    T = CONFIG['num_tokens']
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    D = CONFIG['d_model']

    tokens = jax.random.normal(k1, (T, D), dtype=dtype)
    router_logits = jax.random.normal(k2, (T, E), dtype=dtype)
    expert_Sim_out = jax.random.normal(k3, (E, C, D), dtype=dtype)
    return tokens, router_logits, expert_Sim_out


def workload(tokens, router_logits, expert_Sim_out):
    T, D = tokens.shape
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    orig_dtype = tokens.dtype

    # 1. Top-1 routing in float32
    logits_f32 = router_logits.astype(jnp.float32)
    probs = jax.nn.softmax(logits_f32, axis=-1)
    expert_idx = jnp.argmax(probs, axis=-1)
    gate_weights = jnp.max(probs, axis=-1)

    # 2. Compute position in expert buffer via cumulative sum and apply capacity mask
    expert_mask = jax.nn.one_hot(expert_idx, num_classes=E, dtype=jnp.int32)
    position_in_expert = jnp.cumsum(expert_mask, axis=0) - 1
    slot_idx = jnp.sum(position_in_expert * expert_mask, axis=-1)
    valid_mask = slot_idx < C

    # 3. Dispatch tokens to [E, C, D] buffer (overflow tokens redirect to dummy index E and are dropped)
    safe_expert = jnp.where(valid_mask, expert_idx, E)
    safe_slot = jnp.where(valid_mask, slot_idx, 0)

    dispatched = jnp.zeros((E, C, D), dtype=orig_dtype)
    dispatched = dispatched.at[safe_expert, safe_slot].set(tokens, mode='drop')

    # 4. Combine from expert buffer [E, C, D] back to [T, D] with gate weights and zeroing dropped tokens
    gathered = expert_Sim_out[expert_idx, jnp.minimum(slot_idx, C - 1)]
    effective_weights = jnp.where(valid_mask, gate_weights, 0.0)[:, None]
    combined_f32 = gathered.astype(jnp.float32) * effective_weights
    combined = combined_f32.astype(orig_dtype)

    return dispatched, combined
