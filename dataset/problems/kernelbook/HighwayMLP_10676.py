"""[bf16 variant] HighwayMLP: relu(Linear(x)) * softmax(Linear(x), axis=0) + (1 - gate) * x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HighwayMLP_10676',
    'd0': 16,
    'd1': 16,
    'd2': 64,
    'input_size': 1024,
    'gate_bias': -2.0,
    'softmax_axis': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.key(0)
    kx, kwn, kbn, kwg = jax.random.split(k, 4)
    d = CONFIG['input_size']
    x = jax.random.uniform(kx, (CONFIG['d0'], CONFIG['d1'], CONFIG['d2'], d), dtype)
    w_n = jax.random.normal(kwn, (d, d), dtype) * 0.02
    b_n = jax.random.normal(kbn, (d,), dtype) * 0.02
    w_g = jax.random.normal(kwg, (d, d), dtype) * 0.02
    b_g = jnp.full((d,), CONFIG['gate_bias'], dtype)
    return x, w_n, b_n, w_g, b_g


def workload(x, w_n, b_n, w_g, b_g):
    normal = jax.nn.relu(x @ w_n + b_n)
    gate = jax.nn.softmax(x @ w_g + b_g, axis=CONFIG['softmax_axis'])
    return normal * gate + (1.0 - gate) * x
