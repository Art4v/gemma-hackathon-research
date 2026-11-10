"""Policy MLP: Linear -> tanh -> Linear -> tanh -> Linear (action mean), plus broadcast log_std and exp(log_std)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Policy_4403',
    'batch': 4096,
    'seq': 128,
    'num_inputs': 128,
    'hidden': 64,
    'num_outputs': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (c['batch'], c['seq'], c['num_inputs']), dtype)
    w1 = jax.random.normal(ks[1], (c['num_inputs'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['hidden'], c['hidden']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['hidden'],), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (c['hidden'], c['num_outputs']), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (c['num_outputs'],), dtype) * 0.02
    log_std = jnp.zeros((1, c['num_outputs']), dtype)
    return x, w1, b1, w2, b2, w3, b3, log_std


def workload(x, w1, b1, w2, b2, w3, b3, log_std):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    action_mean = h @ w3 + b3
    action_log_std = jnp.broadcast_to(log_std, action_mean.shape)
    action_std = jnp.exp(action_log_std)
    return action_mean, action_log_std, action_std
