"""[bf16 variant] DownsampleB: AvgPool2d(stride) -> channel concat with zero copies (expand_ratio = nOut // nIn)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DownsampleB_15452',
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
        key, (CONFIG['batch'], CONFIG['n_in'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def workload(x):
    s = CONFIG['stride']
    pooled = lax.reduce_window(
        x, jnp.array(0, x.dtype), lax.add,
        window_dimensions=(1, 1, s, s),
        window_strides=(1, 1, s, s),
        padding='VALID',
    ) / (s * s)
    expand_ratio = CONFIG['n_out'] // CONFIG['n_in']
    return jnp.concatenate([pooled] + [pooled * 0] * (expand_ratio - 1), axis=1)
