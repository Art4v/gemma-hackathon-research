"""[bf16 variant] OneDilate: (1 - x) * 0.5 -> replicate pad -> depthwise all-ones conv (box sum)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OneDilate_11093',
    'batch': 16,
    'channels': 3,
    'height': 1024,
    'width': 1024,
    'kernel_size': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    c, k = CONFIG['channels'], CONFIG['kernel_size']
    x = jax.random.uniform(
        key, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    # Fixed all-ones depthwise filter, PyTorch layout (C, 1, K, K).
    w = jnp.ones((c, 1, k, k), dtype)
    return x, w


def workload(x, w):
    c = x.shape[1]
    k = CONFIG['kernel_size']
    p = (k - 1) // 2
    y = (1.0 - x) * 0.5
    y = jnp.pad(y, ((0, 0), (0, 0), (p, p), (p, p)), mode='edge')
    y = jnp.transpose(y, (0, 2, 3, 1))           # NCHW -> NHWC
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))      # (C,1,K,K) -> (K,K,1,C)
    out = lax.conv_general_dilated(
        y, w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=c,
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(out, (0, 3, 1, 2))      # NHWC -> NCHW
