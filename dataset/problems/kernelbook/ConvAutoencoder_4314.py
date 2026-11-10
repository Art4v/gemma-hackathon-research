"""[bf16 variant] ConvAutoencoder: conv-relu-maxpool x2, then transposed conv-relu, transposed conv-sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvAutoencoder_4314',
    'batch': 16,
    'in_channels': 1,
    'height': 1024,
    'width': 1024,
    'conv_1_channels': 16,
    'conv_2_channels': 4,
    'conv_kernel': 3,
    'conv_stride': 1,
    'conv_padding': 1,
    'pool_window': 2,
    'pool_stride': 2,
    'trans_kernel': 2,
    'trans_stride': 2,
    'trans_padding': 0,
    'trans_output_padding': 0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['conv_kernel']
    t = c['trans_kernel']
    scale = c['weight_scale']
    # Conv2d weights are (out, in, kh, kw); ConvTranspose2d weights are (in, out, kh, kw).
    conv_1_w = jax.random.normal(
        keys[1], (c['conv_1_channels'], c['in_channels'], k, k), dtype) * scale
    conv_1_b = jax.random.normal(keys[2], (c['conv_1_channels'],), dtype) * scale
    conv_2_w = jax.random.normal(
        keys[3], (c['conv_2_channels'], c['conv_1_channels'], k, k), dtype) * scale
    conv_2_b = jax.random.normal(keys[4], (c['conv_2_channels'],), dtype) * scale
    trans_1_w = jax.random.normal(
        keys[5], (c['conv_2_channels'], c['conv_1_channels'], t, t), dtype) * scale
    trans_1_b = jax.random.normal(keys[6], (c['conv_1_channels'],), dtype) * scale
    trans_2_w = jax.random.normal(
        keys[7], (c['conv_1_channels'], c['in_channels'], t, t), dtype) * scale
    trans_2_b = jax.random.normal(keys[8], (c['in_channels'],), dtype) * scale
    return (x, conv_1_w, conv_1_b, conv_2_w, conv_2_b,
            trans_1_w, trans_1_b, trans_2_w, trans_2_b)


def workload(x, conv_1_w, conv_1_b, conv_2_w, conv_2_b,
             trans_1_w, trans_1_b, trans_2_w, trans_2_b):
    c = CONFIG

    def conv(h, w, b):
        p = c['conv_padding']
        s = c['conv_stride']
        out = lax.conv_general_dilated(
            h, w,
            window_strides=(s, s),
            padding=((p, p), (p, p)),
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        )
        return out + b

    def pool(h):
        win = c['pool_window']
        s = c['pool_stride']
        return lax.reduce_window(
            h, -jnp.inf, lax.max, (1, win, win, 1), (1, s, s, 1), 'VALID')

    def trans_conv(h, w, b):
        k = c['trans_kernel']
        s = c['trans_stride']
        lo = k - 1 - c['trans_padding']
        hi = lo + c['trans_output_padding']
        out = lax.conv_general_dilated(
            h, jnp.flip(w, axis=(2, 3)),
            window_strides=(1, 1),
            padding=((lo, hi), (lo, hi)),
            lhs_dilation=(s, s),
            dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
        )
        return out + b

    out = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(conv(out, conv_1_w, conv_1_b))
    out = pool(out)
    out = jax.nn.relu(conv(out, conv_2_w, conv_2_b))
    out = pool(out)
    out = jax.nn.relu(trans_conv(out, trans_1_w, trans_1_b))
    out = jax.nn.sigmoid(trans_conv(out, trans_2_w, trans_2_b))
    return jnp.transpose(out, (0, 3, 1, 2))
