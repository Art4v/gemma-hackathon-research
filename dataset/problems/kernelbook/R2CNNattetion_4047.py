"""[bf16 variant] MaxPool2d(1/2/4) pyramid with ConvTranspose2d upsampling of the 2x and 4x branches, summed."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'R2CNNattetion_4047',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
    'pool1_kernel': 1,
    'pool2_kernel': 2,
    'pool3_kernel': 4,
    'deconv2_kernel': 4,
    'deconv2_stride': 2,
    'deconv2_padding': 1,
    'deconv3_kernel': 6,
    'deconv3_stride': 4,
    'deconv3_padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG['channels']
    k_x, k_w2, k_b2, k_w3, k_b3 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    k2 = CONFIG['deconv2_kernel']
    k3 = CONFIG['deconv3_kernel']
    w2 = jax.random.normal(k_w2, (c, c, k2, k2), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c,), dtype) * 0.02
    w3 = jax.random.normal(k_w3, (c, c, k3, k3), dtype) * 0.02
    b3 = jax.random.normal(k_b3, (c,), dtype) * 0.02
    return x, w2, b2, w3, b3


def _max_pool(x, kernel):
    window = (1, kernel, kernel, 1)
    return lax.reduce_window(x, -jnp.inf, lax.max, window, window, 'VALID')


def _deconv(x, w, b, kernel, stride, padding):
    pad = kernel - 1 - padding
    y = lax.conv_general_dilated(
        x,
        jnp.flip(w, axis=(2, 3)),
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    return y + b


def workload(x, w2, b2, w3, b3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    x1 = _max_pool(h, CONFIG['pool1_kernel'])
    x2 = _max_pool(h, CONFIG['pool2_kernel'])
    x3 = _max_pool(h, CONFIG['pool3_kernel'])
    x2 = _deconv(x2, w2, b2, CONFIG['deconv2_kernel'], CONFIG['deconv2_stride'], CONFIG['deconv2_padding'])
    x3 = _deconv(x3, w3, b3, CONFIG['deconv3_kernel'], CONFIG['deconv3_stride'], CONFIG['deconv3_padding'])
    out = x1 + x2 + x3
    return jnp.transpose(out, (0, 3, 1, 2))
