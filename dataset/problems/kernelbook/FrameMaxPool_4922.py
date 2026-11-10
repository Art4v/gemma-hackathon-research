"""[bf16 variant] Conv1d (kernel 1) with bias -> ReLU -> MaxPool1d over the temporal axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FrameMaxPool_4922',
    'batch': 16,
    'input_size': 512,
    'hidden_size': 512,
    'length': 4096,
    'kernel_size': 1,
    'conv_stride': 1,
    'stride': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['input_size'], CONFIG['length']), dtype
    )
    w = jax.random.normal(
        k_w, (CONFIG['hidden_size'], CONFIG['input_size'], CONFIG['kernel_size']), dtype
    ) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['hidden_size'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    unbatched = x.ndim == 2
    if unbatched:
        x = x[None]
    h = jnp.transpose(x, (0, 2, 1))
    h = lax.conv_general_dilated(
        h,
        w,
        window_strides=(CONFIG['conv_stride'],),
        padding='VALID',
        dimension_numbers=('NWC', 'OIW', 'NWC'),
    )
    h = h + b
    h = jnp.maximum(h, 0)
    s = CONFIG['stride']
    h = lax.reduce_window(h, -jnp.inf, lax.max, (1, s, 1), (1, s, 1), 'VALID')
    h = jnp.transpose(h, (0, 2, 1))
    if unbatched:
        h = h[0]
    return h
