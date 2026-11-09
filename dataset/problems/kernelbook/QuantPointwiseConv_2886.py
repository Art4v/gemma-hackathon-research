"""QuantPointwiseConv: identity quant stubs, transpose(1,2), Linear on last axis, transpose back, identity dequant."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QuantPointwiseConv_2886',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'in_features': 512,
    'out_features': 512,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['in_features']),
        dtype,
    )
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # Quant/DeQuant stubs are identity on an unconverted model.
    xt = jnp.swapaxes(x, 1, 2)
    y = jnp.matmul(xt, w) + b
    return jnp.swapaxes(y, 1, 2)
