"""GRUCell variant: linear(x, W_ih, b) + linear(h, W_hh_rz) -> sigmoid gates -> relu candidate with linear(h*r, W_hh) -> interpolate."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GRUCell_17050',
    'batch': 1024,
    'input_size': 2048,
    'hidden_size': 2048,
}


def create_inputs(dtype=jnp.float32):
    b, i, h = CONFIG['batch'], CONFIG['input_size'], CONFIG['hidden_size']
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(keys[0], (b, i), dtype)
    hx = jax.random.uniform(keys[1], (b, h), dtype)
    w_ih = jax.random.normal(keys[2], (i, 3 * h), dtype) * 0.02
    w_hh_rz = jax.random.normal(keys[3], (h, 2 * h), dtype) * 0.02
    w_hh = jax.random.normal(keys[4], (h, h), dtype) * 0.02
    bias = jax.random.normal(keys[5], (3 * h,), dtype) * 0.02
    return x, hx, w_ih, w_hh_rz, w_hh, bias


def workload(x, h, w_ih, w_hh_rz, w_hh, bias):
    hs = h.shape[1]
    hp = lax.Precision.HIGHEST
    ih = jnp.dot(x, w_ih, precision=hp) + bias
    hh_rz = jnp.dot(h, w_hh_rz, precision=hp)
    r = jax.nn.sigmoid(ih[:, :hs] + hh_rz[:, :hs])
    i = jax.nn.sigmoid(ih[:, hs:2 * hs] + hh_rz[:, hs:])
    hhr = jnp.dot(h * r, w_hh, precision=hp)
    n = jax.nn.relu(ih[:, 2 * hs:] + hhr)
    return (1 - i) * n + i * h
