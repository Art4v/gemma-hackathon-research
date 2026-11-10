"""[bf16 variant] Vector-neuron max pool: linear direction map over feature axis, channel-wise dot product, argmax over samples, gather."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VNMaxPool_14508',
    'batch': 8,
    'n_feat': 256,
    'n_channels': 128,
    'n_samples': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['n_feat'], CONFIG['n_channels'], CONFIG['n_samples']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['n_feat'], CONFIG['n_feat']), dtype) * CONFIG['weight_scale']
    return x, w


def workload(x, w):
    d = jnp.swapaxes(jnp.swapaxes(x, 1, -1) @ w, 1, -1)
    dotprod = jnp.sum(x * d, axis=2, keepdims=True)
    idx = jnp.argmax(dotprod, axis=-1)
    idx = jnp.broadcast_to(idx, x.shape[:-1])[..., None]
    return jnp.squeeze(jnp.take_along_axis(x, idx, axis=-1), axis=-1)
