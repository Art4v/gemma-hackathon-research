"""[bf16 variant] SpatialAttentionLayer: channel max/mean pool -> conv2d(2->1, 3x3) -> sigmoid -> x * scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttentionLayer_18244',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    w = jax.random.normal(kw, (1, 2, k, k), dtype) * 0.02  # OIHW, like PyTorch
    return x, w


def workload(x, w):
    k = w.shape[-1]
    pad = k // 2
    pooled = jnp.concatenate(
        [jnp.max(x, axis=1, keepdims=True), jnp.mean(x, axis=1, keepdims=True)], axis=1)
    pooled_nhwc = jnp.transpose(pooled, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        pooled_nhwc, w_hwio, window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    scale = jax.nn.sigmoid(jnp.transpose(out, (0, 3, 1, 2)))
    return x * scale
