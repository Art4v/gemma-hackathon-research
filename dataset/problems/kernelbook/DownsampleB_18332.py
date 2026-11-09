"""DownsampleB: AvgPool2d(stride) -> channel concat with zeroed copies (expand_ratio = nOut // nIn)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DownsampleB_18332',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'n_in': 128,
    'n_out': 128,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    s = CONFIG['stride']
    expand_ratio = CONFIG['n_out'] // CONFIG['n_in']
    # AvgPool2d(s): kernel = stride = s, no padding, floor mode -> VALID window.
    pooled = lax.reduce_window(
        x, jnp.array(0, x.dtype), lax.add,
        window_dimensions=(1, 1, s, s),
        window_strides=(1, 1, s, s),
        padding='VALID',
    ) / (s * s)
    # x * 0 (not zeros) to match torch's mul(0) NaN/inf propagation.
    return jnp.concatenate([pooled] + [pooled * 0] * (expand_ratio - 1), axis=1)
