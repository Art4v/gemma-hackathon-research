"""[bf16 variant] Channel squeeze-excitation: global avg pool -> linear -> relu -> linear -> sigmoid -> scale input channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'cSEmodule_2356',
    'batch': 8,
    'in_channel': 128,
    'reduced_channel': 64,
    'height': 256,
    'width': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    c = CONFIG['in_channel']
    r = CONFIG['reduced_channel']
    h = CONFIG['height']
    w = CONFIG['width']
    scale = CONFIG['weight_scale']
    k_x, k_wd, k_bd, k_wu, k_bu = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    w_down = jax.random.normal(k_wd, (c, r), dtype) * scale
    b_down = jax.random.normal(k_bd, (r,), dtype) * scale
    w_up = jax.random.normal(k_wu, (r, c), dtype) * scale
    b_up = jax.random.normal(k_bu, (c,), dtype) * scale
    return x, w_down, b_down, w_up, b_up


def workload(x, w_down, b_down, w_up, b_up):
    b, c = x.shape[0], x.shape[1]
    pooled = jnp.mean(x, axis=(2, 3))
    hidden = jax.nn.relu(pooled @ w_down + b_down)
    attention = jax.nn.sigmoid(hidden @ w_up + b_up)
    attention = attention.reshape(b, c, 1, 1)
    return attention * x
