"""[bf16 variant] TransitionModel: sigmoid -> clamped log -> roll/shift leaving -> stack -> logsumexp -> length-mask fill."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransitionModel_15636',
    'batch': 1024,
    'num_states': 32768,
    'eps': 0.0001,
    'shift': 1,
    'state_axis': 1,
    'stack_axis': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    k_alpha, k_transition, k_lengths = jax.random.split(jax.random.key(0), 3)
    batch = CONFIG['batch']
    num_states = CONFIG['num_states']
    log_alpha_scaled = jax.random.normal(k_alpha, (batch, num_states), dtype)
    transition_vector = jax.random.normal(k_transition, (num_states,), dtype)
    state_lengths = jax.random.randint(k_lengths, (batch,), 1, num_states + 1, jnp.int32)
    return log_alpha_scaled, transition_vector, state_lengths


def workload(log_alpha_scaled, transition_vector, state_lengths):
    eps = CONFIG['eps']
    state_axis = CONFIG['state_axis']
    stack_axis = CONFIG['stack_axis']
    t_max = log_alpha_scaled.shape[state_axis]

    transition_probability = jax.nn.sigmoid(transition_vector)
    staying_probability = jax.nn.sigmoid(-transition_vector)
    log_staying_probability = jnp.log(jnp.maximum(staying_probability, eps))
    log_transition_probability = jnp.log(jnp.maximum(transition_probability, eps))

    staying = log_alpha_scaled + log_staying_probability
    leaving = log_alpha_scaled + log_transition_probability
    leaving = jnp.roll(leaving, CONFIG['shift'], axis=state_axis)
    leaving = leaving.at[:, 0].set(-jnp.inf)

    positions = jnp.broadcast_to(
        jnp.arange(t_max, dtype=log_alpha_scaled.dtype),
        (state_lengths.shape[0], t_max),
    )
    not_state_lengths_mask = jnp.logical_not(
        positions < jnp.expand_dims(state_lengths, state_axis)
    )

    stacked = jnp.stack((staying, leaving), axis=stack_axis)
    max_value = jnp.max(stacked, axis=stack_axis)
    all_inf = max_value == -jnp.inf
    safe_max = jnp.where(all_inf, 0.0, max_value)
    summed = jnp.sum(
        jnp.exp(stacked - jnp.expand_dims(safe_max, stack_axis)), axis=stack_axis
    )
    out = jnp.log(jnp.where(all_inf, 1.0, summed)) + jnp.where(all_inf, -jnp.inf, safe_max)

    return jnp.where(not_state_lengths_mask, -jnp.inf, out)
