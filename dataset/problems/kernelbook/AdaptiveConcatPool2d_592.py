"""[bf16 variant] AdaptiveConcatPool2d: concat(global max pool, global avg pool) over channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdaptiveConcatPool2d_592',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'output_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    # ponytail: adaptive pool with output_size=1 is a global reduction; general sizes unused here.
    assert CONFIG['output_size'] == 1
    mp = jnp.max(x, axis=(2, 3), keepdims=True)
    ap = jnp.mean(x, axis=(2, 3), keepdims=True)
    return jnp.concatenate([mp, ap], axis=1)
