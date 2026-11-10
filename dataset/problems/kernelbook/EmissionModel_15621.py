"""Diagonal Normal log-likelihood of observations against state means, masked by state lengths and summed over features."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EmissionModel_15621',
    'batch': 128,
    'hidden_state': 512,
    'feature_dim': 512,
    'std_min': 0.1,
    'std_max': 1.0,
    'min_state_length': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_means, k_stds, k_lengths = jax.random.split(key, 4)
    batch = CONFIG['batch']
    hidden_state = CONFIG['hidden_state']
    feature_dim = CONFIG['feature_dim']
    x_t = jax.random.uniform(k_x, (batch, feature_dim), dtype)
    means = jax.random.uniform(k_means, (batch, hidden_state, feature_dim), dtype)
    stds = jax.random.uniform(
        k_stds, (feature_dim,), dtype,
        minval=CONFIG['std_min'], maxval=CONFIG['std_max'],
    )
    state_lengths = jax.random.randint(
        k_lengths, (batch,), CONFIG['min_state_length'], hidden_state + 1, jnp.int32,
    )
    return x_t, means, stds, state_lengths


def workload(x_t, means, stds, state_lengths):
    t_max = means.shape[1]
    x = jnp.expand_dims(x_t, 1)
    var = stds ** 2
    log_prob = -((x - means) ** 2) / (2 * var) - jnp.log(stds) - jnp.log(jnp.sqrt(2.0 * jnp.pi))
    positions = jnp.broadcast_to(
        jnp.arange(t_max, dtype=x.dtype), (state_lengths.shape[0], t_max),
    )
    mask = jnp.expand_dims(positions < jnp.expand_dims(state_lengths, 1), 2)
    return jnp.sum(log_prob * mask, axis=2)
