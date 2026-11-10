"""PolicyNetwork forward: linear-relu-linear-relu, then mean and clamped log_std linear heads."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PolicyNetwork_17881',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'num_inputs': 256,
    'hidden_size': 1024,
    'num_actions': 256,
    'log_std_min': -20.0,
    'log_std_max': 2.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    n_in = CONFIG['num_inputs']
    hidden = CONFIG['hidden_size']
    n_act = CONFIG['num_actions']
    scale = CONFIG['weight_scale']
    state = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], n_in), dtype)
    w1 = jax.random.normal(keys[1], (n_in, hidden), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden, hidden), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden,), dtype) * scale
    w_mean = jax.random.normal(keys[5], (hidden, n_act), dtype) * scale
    b_mean = jax.random.normal(keys[6], (n_act,), dtype) * scale
    w_log_std = jax.random.normal(keys[7], (hidden, n_act), dtype) * scale
    b_log_std = jax.random.normal(keys[8], (n_act,), dtype) * scale
    return (state, w1, b1, w2, b2, w_mean, b_mean, w_log_std, b_log_std)


def workload(state, w1, b1, w2, b2, w_mean, b_mean, w_log_std, b_log_std):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    mean = x @ w_mean + b_mean
    log_std = x @ w_log_std + b_log_std
    log_std = jnp.clip(log_std, CONFIG['log_std_min'], CONFIG['log_std_max'])
    return mean, log_std
