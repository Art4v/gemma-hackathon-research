"""FusedDownsample: equalized-lr weight scale -> zero pad -> 2x2 box blur of kernel -> stride-2 conv2d -> per-channel bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FusedDownsample_1654',
    'batch': 8,
    'in_channel': 128,
    'out_channel': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 2,
    'padding': 0,
    'blur_pad': 1,
    'blur_divisor': 4.0,
    'gain': 2.0,
    'init_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channel'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jax.random.normal(
        k_w,
        (CONFIG['out_channel'], CONFIG['in_channel'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['init_scale']
    bias = jax.random.normal(k_b, (CONFIG['out_channel'],), dtype) * CONFIG['init_scale']
    return x, weight, bias


def workload(x, weight, bias):
    fan_in = weight.shape[1] * weight.shape[2] * weight.shape[3]
    multiplier = (CONFIG['gain'] / fan_in) ** 0.5

    p = CONFIG['blur_pad']
    w = jnp.pad(weight * multiplier, ((0, 0), (0, 0), (p, p), (p, p)))
    w = (
        w[:, :, 1:, 1:]
        + w[:, :, :-1, 1:]
        + w[:, :, 1:, :-1]
        + w[:, :, :-1, :-1]
    ) / CONFIG['blur_divisor']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    pad = CONFIG['padding']
    out = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(CONFIG['stride'], CONFIG['stride']),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + bias
    return jnp.transpose(out, (0, 3, 1, 2))
