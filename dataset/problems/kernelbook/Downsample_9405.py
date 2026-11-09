"""[bf16 variant] Downsample_9405: AvgPool2d(stride) followed by channel concat with zeroed copies."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Downsample_9405',
    'batch': 8,
    'n_in': 128,
    'n_out': 128,
    'height': 256,
    'width': 256,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['n_in'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    s = CONFIG['stride']
    ratio = CONFIG['n_out'] // CONFIG['n_in']
    # AvgPool2d(s): kernel=s, stride=s, padding=0, floor mode -> VALID window mean.
    summed = lax.reduce_window(
        x, jnp.array(0, x.dtype), lax.add,
        window_dimensions=(1, 1, s, s),
        window_strides=(1, 1, s, s),
        padding='VALID',
    )
    y = summed / jnp.asarray(s * s, x.dtype)
    # y * 0 (not zeros_like) mirrors x.mul(0), preserving NaN/Inf propagation.
    return jnp.concatenate([y] + [y * 0] * (ratio - 1), axis=1)
