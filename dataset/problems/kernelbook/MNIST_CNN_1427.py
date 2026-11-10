"""[bf16 variant] MNIST_CNN: (Conv2d -> ReLU -> GroupNorm) x 4 -> global average pool -> flatten."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MNIST_CNN_1427',
    'batch': 16,
    'in_channels': 4,
    'height': 512,
    'width': 512,
    'conv1_channels': 64,
    'conv_channels': 128,
    'kernel_size': 3,
    'padding': 1,
    'strides': (1, 2, 1, 1),
    'num_groups': 8,
    'eps': 1e-5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel_size']
    c0 = c['in_channels']
    c1 = c['conv1_channels']
    c2 = c['conv_channels']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], c0, c['height'], c['width']), dtype)
    scale = c['weight_scale']
    conv1_w = jax.random.normal(keys[1], (c1, c0, k, k), dtype) * scale
    conv1_b = jax.random.normal(keys[2], (c1,), dtype) * scale
    conv2_w = jax.random.normal(keys[3], (c2, c1, k, k), dtype) * scale
    conv2_b = jax.random.normal(keys[4], (c2,), dtype) * scale
    conv3_w = jax.random.normal(keys[5], (c2, c2, k, k), dtype) * scale
    conv3_b = jax.random.normal(keys[6], (c2,), dtype) * scale
    conv4_w = jax.random.normal(keys[7], (c2, c2, k, k), dtype) * scale
    conv4_b = jax.random.normal(keys[8], (c2,), dtype) * scale
    gn0_scale = jnp.ones((c1,), dtype)
    gn0_shift = jnp.zeros((c1,), dtype)
    gn1_scale = jnp.ones((c2,), dtype)
    gn1_shift = jnp.zeros((c2,), dtype)
    gn2_scale = jnp.ones((c2,), dtype)
    gn2_shift = jnp.zeros((c2,), dtype)
    gn3_scale = jnp.ones((c2,), dtype)
    gn3_shift = jnp.zeros((c2,), dtype)
    return (
        x,
        conv1_w, conv1_b,
        conv2_w, conv2_b,
        conv3_w, conv3_b,
        conv4_w, conv4_b,
        gn0_scale, gn0_shift,
        gn1_scale, gn1_shift,
        gn2_scale, gn2_shift,
        gn3_scale, gn3_shift,
    )


def _conv(x, w, b, stride):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _group_norm(x, scale, shift):
    n, h, w, c = x.shape
    groups = CONFIG['num_groups']
    g = x.reshape(n, h, w, groups, c // groups)
    axes = (1, 2, 4)
    mean = jnp.mean(g, axis=axes, keepdims=True)
    centered = g - mean
    var = jnp.mean(centered * centered, axis=axes, keepdims=True)
    out = centered / jnp.sqrt(var + CONFIG['eps'])
    return out.reshape(n, h, w, c) * scale + shift


def workload(
    x,
    conv1_w, conv1_b,
    conv2_w, conv2_b,
    conv3_w, conv3_b,
    conv4_w, conv4_b,
    gn0_scale, gn0_shift,
    gn1_scale, gn1_shift,
    gn2_scale, gn2_shift,
    gn3_scale, gn3_shift,
):
    s1, s2, s3, s4 = CONFIG['strides']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = _group_norm(jax.nn.relu(_conv(y, conv1_w, conv1_b, s1)), gn0_scale, gn0_shift)
    y = _group_norm(jax.nn.relu(_conv(y, conv2_w, conv2_b, s2)), gn1_scale, gn1_shift)
    y = _group_norm(jax.nn.relu(_conv(y, conv3_w, conv3_b, s3)), gn2_scale, gn2_shift)
    y = _group_norm(jax.nn.relu(_conv(y, conv4_w, conv4_b, s4)), gn3_scale, gn3_shift)
    return jnp.mean(y, axis=(1, 2))
