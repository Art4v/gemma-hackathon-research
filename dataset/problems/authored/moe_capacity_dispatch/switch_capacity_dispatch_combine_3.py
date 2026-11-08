"""Switch Transformer capacity-limited top-1 token dispatch with overflow drop and weighted combine."""

import jax
import jax.numpy as jnp

CONFIG = {
    "name": "switch_capacity_dispatch_combine",
    "num_tokens": 65536,
    "num_experts": 32,
    "capacity": 2560,
    "hidden_size": 4096,
    "eps": 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    num_tokens = CONFIG["num_tokens"]
    num_experts = CONFIG["num_experts"]
    capacity = CONFIG["capacity"]
    hidden_size = CONFIG["hidden_size"]

    tokens = jax.random.normal(k0, (num_tokens, hidden_size), dtype=dtype)
    router_logits = jax.random.normal(k1, (num_tokens, num_experts), dtype=dtype)
    expert_outputs = jax.random.normal(
        k2, (num_experts, capacity, hidden_size), dtype=dtype
    )
    return tokens, router_logits, expert_outputs


def workload(tokens, router_logits, expert_outputs):
    num_tokens = CONFIG["num_tokens"]
    num_experts = CONFIG["num_experts"]
    capacity = CONFIG["capacity"]
    hidden_size = CONFIG["hidden_size"]

    logits_f32 = router_logits.astype(jnp.float32)
    router_probs = jax.nn.softmax(logits_f32, axis=-1)
    expert_idx = jnp.argmax(logits_f32, axis=-1)
    gate_score = jnp.take_along_axis(
        router_probs, expert_idx[:, None], axis=-1
    ).squeeze(-1)

    expert_mask = jax.nn.one_hot(expert_idx, num_experts, dtype=jnp.int32)
    position_in_expert = jnp.cumsum(expert_mask, axis=0) - 1
    slot_idx = jnp.sum(position_in_expert * expert_mask, axis=-1)

    valid_mask = slot_idx < capacity
    safe_expert = jnp.where(valid_mask, expert_idx, num_experts)
    safe_slot = jnp.where(valid_mask, slot_idx, capacity)

    padded_buffer = jnp.zeros(
        (num_experts + 1, capacity + 1, hidden_size), dtype=tokens.dtype
    )
    padded_buffer = padded_buffer.at[safe_expert, safe_slot].set(tokens)
    dispatched_tokens = padded_buffer[:num_experts, :capacity, :]

    safe_gather_slot = jnp.where(valid_mask, slot_idx, 0)
    gathered = expert_outputs[expert_idx, safe_gather_slot]
    scale = jnp.where(valid_mask, gate_score, 0.0)[:, None]
    combined_tokens = (gathered.astype(jnp.float32) * scale).astype(tokens.dtype)

    return dispatched_tokens, combined_tokens
