"""Channel concat of two images followed by a 1x1 Conv2d (2*C -> 2) with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MockOpticalFlowModel_16159',
    'batch': 16,
    'img_channels': 128,
    'height': 128,
    'width': 128,
    'out_channels': 2,
    'kernel_size': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    b, c = CONFIG['batch'], CONFIG['img_channels']
    h, w = CONFIG['height'], CONFIG['width']
    ks, oc = CONFIG['kernel_size'], CONFIG['out_channels']
    img1 = jax.random.uniform(k1, (b, c, h, w), dtype)
    img2 = jax.random.uniform(k2, (b, c, h, w), dtype)
    weight = jax.random.normal(k3, (ks, ks, 2 * c, oc), dtype) * 0.02  # HWIO
    bias = jax.random.normal(k4, (oc,), dtype) * 0.02
    return img1, img2, weight, bias


def workload(img1, img2, weight, bias):
    x = jnp.concatenate([img1, img2], axis=1)  # dim=-3 of NCHW
    x = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x, weight, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + bias
    return jnp.transpose(y, (0, 3, 1, 2))
