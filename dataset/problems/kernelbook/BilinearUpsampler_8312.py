"""[bf16 variant] ReplicationPad2d -> ConvTranspose2d (width, bilinear) -> ConvTranspose2d (height, bilinear) upsampler."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BilinearUpsampler_8312',
    'batch': 8,
    'channels': 1,
    'height': 2048,
    'width': 2048,
    'scale': 2,
    'ksize': 4,
    'total_pad': 3,
    'replication_pad': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    c = CONFIG['channels']
    scale = CONFIG['scale']
    ksize = CONFIG['ksize']
    x = jax.random.uniform(
        key, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype
    )
    k_idx = jnp.arange(ksize, dtype=dtype) + 0.5
    k_coord = k_idx / scale - ksize * 0.5 / scale
    k_weight = jnp.clip(1.0 - jnp.abs(k_coord), 0.0, None)
    w_x = jnp.broadcast_to(k_weight[None, None, None, :], (c, 1, 1, ksize))
    w_y = jnp.broadcast_to(k_weight[None, None, :, None], (c, 1, ksize, 1))
    return x, w_x, w_y


def _conv_transpose(x, w, stride, pad):
    # x is NHWC; w is PyTorch ConvTranspose2d layout (groups, 1, h, w).
    kernel = jnp.flip(jnp.transpose(w, (2, 3, 1, 0)), axis=(0, 1))
    padding = [
        (k - 1 - p, k - 1 - p) for k, p in zip(kernel.shape[:2], pad)
    ]
    return lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=padding,
        lhs_dilation=stride,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=x.shape[-1],
    )


def workload(x, w_x, w_y):
    scale = CONFIG['scale']
    total_pad = CONFIG['total_pad']
    rp = CONFIG['replication_pad']
    y = jnp.pad(x, ((0, 0), (0, 0), (rp, rp), (rp, rp)), mode='edge')
    y = jnp.transpose(y, (0, 2, 3, 1))
    y = _conv_transpose(y, w_x, (1, scale), (0, total_pad))
    y = _conv_transpose(y, w_y, (scale, 1), (total_pad, 0))
    return jnp.transpose(y, (0, 3, 1, 2))
