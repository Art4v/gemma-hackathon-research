"""[bf16 variant] FC_Q: two 3-layer MLP heads (linear+relu chain) on the last axis, log_softmax over axis 1 for the i-head."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FC_Q_8974',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'state_dim': 128,
    'hidden': 256,
    'num_actions': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 13)
    s, h, a = CONFIG['state_dim'], CONFIG['hidden'], CONFIG['num_actions']
    state = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], s), dtype)
    shapes = [(s, h), (h,), (h, h), (h,), (h, a), (a,)]
    params = [jax.random.normal(k, shp, dtype) * 0.02
              for k, shp in zip(keys[1:], shapes + shapes)]
    return (state, *params)


def workload(state, q1_w, q1_b, q2_w, q2_b, q3_w, q3_b,
             i1_w, i1_b, i2_w, i2_b, i3_w, i3_b):
    q = jax.nn.relu(state @ q1_w + q1_b)
    q = jax.nn.relu(q @ q2_w + q2_b)
    i = jax.nn.relu(state @ i1_w + i1_b)
    i = jax.nn.relu(i @ i2_w + i2_b)
    i = jax.nn.relu(i @ i3_w + i3_b)
    return q @ q3_w + q3_b, jax.nn.log_softmax(i, axis=CONFIG['softmax_axis']), i
