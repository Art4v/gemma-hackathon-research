"""Dueling Q-network: advantage MLP and value MLP (Linear+ReLU x2, Linear), out = adv - mean(adv) + value."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DuelingQNetwork_10610',
    'batch': 16,
    'seq': 2048,
    'state_size': 1024,
    'hidden_advantage': [1024, 1024],
    'hidden_state_value': [1024, 1024],
    'action_size': 128,
}


def _tower_params(key, dims, dtype):
    params = []
    keys = jax.random.split(key, len(dims) - 1)
    for k, d_in, d_out in zip(keys, dims[:-1], dims[1:]):
        params.append(jax.random.normal(k, (d_in, d_out), dtype) * 0.02)
        params.append(jnp.zeros((d_out,), dtype))
    return params


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_adv, k_val = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['seq'], c['state_size']), dtype)
    adv = _tower_params(k_adv, [c['state_size']] + c['hidden_advantage'] + [c['action_size']], dtype)
    val = _tower_params(k_val, [c['state_size']] + c['hidden_state_value'] + [1], dtype)
    return (x, *adv, *val)


def _mlp(h, params):
    n = len(params) // 2
    for i in range(n):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < n - 1:
            h = jax.nn.relu(h)
    return h


def workload(x, *params):
    n_adv = 2 * (len(CONFIG['hidden_advantage']) + 1)
    advantage = _mlp(x, params[:n_adv])
    value = _mlp(x, params[n_adv:])
    # Global mean over the whole advantage tensor, matching torch's advantage.mean().
    return advantage - jnp.mean(advantage) + value
