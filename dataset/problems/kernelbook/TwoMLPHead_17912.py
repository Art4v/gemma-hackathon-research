"""[bf16 variant] TwoMLPHead: flatten -> linear -> relu -> linear -> relu."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TwoMLPHead_17912',
    'batch': 2048,
    'in_channels': 8192,
    'out_channels': 2048,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 5)
    batch = CONFIG['batch']
    in_channels = CONFIG['in_channels']
    out_channels = CONFIG['out_channels']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(keys[0], (batch, in_channels), dtype)
    w6 = jax.random.normal(keys[1], (in_channels, out_channels), dtype) * scale
    b6 = jax.random.normal(keys[2], (out_channels,), dtype) * scale
    w7 = jax.random.normal(keys[3], (out_channels, out_channels), dtype) * scale
    b7 = jax.random.normal(keys[4], (out_channels,), dtype) * scale
    return x, w6, b6, w7, b7


def workload(x, w6, b6, w7, b7):
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.relu(x @ w6 + b6)
    x = jax.nn.relu(x @ w7 + b7)
    return x
