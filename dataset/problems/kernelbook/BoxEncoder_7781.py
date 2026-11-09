"""BoxEncoder: Linear -> tanh -> Linear -> tanh -> Linear -> tanh over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BoxEncoder_7781',
    'batch': 8,
    'd1': 32,
    'd2': 256,
    'box_size': 1024,
    'feature_size': 1024,
    'hidden_size': 2048,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], c['d1'], c['d2'], c['box_size']), dtype)
    w_enc = jax.random.normal(keys[1], (c['box_size'], c['feature_size']), dtype) * 0.02
    b_enc = jax.random.normal(keys[2], (c['feature_size'],), dtype) * 0.02
    w_in = jax.random.normal(keys[3], (c['feature_size'], c['hidden_size']), dtype) * 0.02
    b_in = jax.random.normal(keys[4], (c['hidden_size'],), dtype) * 0.02
    w_out = jax.random.normal(keys[5], (c['hidden_size'], c['feature_size']), dtype) * 0.02
    b_out = jax.random.normal(keys[6], (c['feature_size'],), dtype) * 0.02
    return x, w_enc, b_enc, w_in, b_in, w_out, b_out


def workload(x, w_enc, b_enc, w_in, b_in, w_out, b_out):
    h = jnp.tanh(x @ w_enc + b_enc)
    h = jnp.tanh(h @ w_in + b_in)
    return jnp.tanh(h @ w_out + b_out)
