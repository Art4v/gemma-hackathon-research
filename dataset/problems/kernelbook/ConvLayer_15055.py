"""ConvLayer: softmax(weight, dim=1) channel-weighted sum of edges over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ConvLayer_15055',
    'batch': 16,          # must equal out_channels (PyTorch broadcast of (O,I,1,1) against (B,C,H,W))
    'out_channels': 16,
    'in_channels': 128,   # must equal input channel dim
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    edges = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    weight = jax.random.normal(
        k_w, (CONFIG['out_channels'], CONFIG['in_channels'], 1, 1), dtype) * 0.02
    return edges, weight


def workload(edges, weight):
    w = jax.nn.softmax(weight, axis=1)
    return jnp.sum(edges * w, axis=1)
