"""Switch Transformer capacity-limited top-1 token dispatch with overflow drop and gated combine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'switch_capacity_dispatch_combine',
    'num_tokens': 32768,
    'num_experts': 32,
    'capacity': 1280,
    'hidden_dim': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_tokens, k_expert_out, k_logits = jax.random.split(key, 3)
    num_tokens = CONFIG['num_tokens']
    num_experts = CONFIG['num_experts']
    capacity = CONFIG['capacity']
    hidden_dim = CONFIG['hidden_dim']

    tokens = jax.random.normal(k_tokens, (num_tokens, hidden_dim), dtype=dtype)
    expert_Buffer_transformed = jax.random.normal(
        k_expert_out, (num_experts, capacity, hidden_dim), dtype=dtype
    )
    router_logits = jax.random.normal(
        k_logits, (num_tokens, num_experts), dtype=jnp.float32
    )
    return tokens, expert_Buffer_transformed, router_logits


def workload(tokens, expert_buffer_transformed, router_logits):
    num_tokens = CONFIG['num_tokens']
    num_experts = CONFIG['num_experts']
    capacity = CONFIG['capacity']
    hidden_dim = CONFIG['hidden_dim']
    orig_dtype = tokens.dtype

    router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    expert_index = jnp.argmax(router_probs, axis=-1)
    expert_gate = jnp.max(router_probs, axis=-1)

    expert_mask = jax.nn.one_hot(expert_index, num_experts, dtype=jnp.int32)
    position_in_expert = jnp.cumsum(expert_mask, axis=0) - 1
    slot_index = jnp.sum(position_in_expert * expert_mask, axis=-1)

    valid_mask = slot_index < capacity
    safe_slot = jnp.where(valid_mask, slot_index, capacity)

    padded_buffer = jnp.zeros(
        (num_experts, capacity + 1, hidden_dim), dtype=orig_dtype
    )
    masked_tokens = jnp.where(valid_mask[:, None], tokens, jnp.zeros_like(tokens))
    padded_buffer = padded_buffer.at[expert_index, safe_slot].set(masked_tokens)
    dispatched_buffer = padded_buffer[:, :capacity, :]

    padded_expert_out = jnp.zeros(
        (num_experts, capacity + 1, hidden_dim), dtype=orig_dtype
    )
    padded_expert_out = padded_expert_out.at[:, :capacity, :].set(
        expert_buffer_transformed
    )
    gathered_tokens = padded_expert_out[expert_index, safe_slot]

    effective_gate = jnp.where(valid_mask, expert_gate, 0.0)
    combined_tokens = (
        gathered_tokens.astype(jnp.float32) * effective_gate[:, None]
    ).astype(orig_dtype)

    tokens_per_expert = jnp.mean(expert_mask.astype(jnp.float32), axis=0)
    prob_per_expert = jnp.mean(router_probs, axis=0)
    aux_loss = (
        jnp.sum(tokens_per_expert * prob_per_expert)
        * jnp.array(num_experts, dtype=jnp.float32)
    )

    return dispatched_buffer, combined_tokens, aux_loss
