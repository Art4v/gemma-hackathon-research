"""[bf16 variant] Actor MLP: Linear -> tanh -> Linear -> tanh -> Linear -> softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_4242',
    'batch': 16,
    'd1': 128,
    'd2': 128,
    'obs_dim': 128,
    'hidden': 64,
    'ac_dim': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['d1'], c['d2'], c['obs_dim']), dtype)
    w1 = jax.random.normal(keys[1], (c['obs_dim'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['hidden'], c['hidden']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['hidden'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['hidden'], c['ac_dim']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['ac_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    # Matches torch.softmax(..., dim=1) on the 4D tensor, which is not the action axis.
    return jax.nn.softmax(h @ w3 + b3, axis=CONFIG['softmax_axis'])
