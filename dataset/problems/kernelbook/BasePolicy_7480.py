"""[bf16 variant] BasePolicy MLP: Linear -> leaky_relu -> Linear -> leaky_relu -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BasePolicy_7480',
    'batch_shape': (32, 32, 128),
    'input_dim': 256,
    'hidden_dim': 512,
    'out_dim': 256,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    d_in, d_h, d_out = CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['out_dim']
    x = jax.random.uniform(keys[0], CONFIG['batch_shape'] + (d_in,), dtype)
    w1 = jax.random.normal(keys[1], (d_in, d_h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (d_h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (d_h, d_h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (d_h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (d_h, d_out), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (d_out,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    slope = CONFIG['negative_slope']
    h1 = jax.nn.leaky_relu(x @ w1 + b1, negative_slope=slope)
    h2 = jax.nn.leaky_relu(h1 @ w2 + b2, negative_slope=slope)
    return h2 @ w3 + b3
