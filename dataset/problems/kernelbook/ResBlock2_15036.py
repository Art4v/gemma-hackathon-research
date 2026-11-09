"""[bf16 variant] ResBlock2: conv1x1 -> InstanceNorm -> ReLU -> conv3x3 -> InstanceNorm -> ReLU -> conv1x1 -> InstanceNorm -> +residual -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResBlock2_15036',
    'batch': 8,
    'input_feature': 128,
    'planes': 128,
    'height': 128,
    'width': 128,
    'dilated': 1,
    'group': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    c_in, planes = CONFIG['input_feature'], CONFIG['planes']
    g = CONFIG['group']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k1, (1, 1, c_in // g, planes), dtype) * 0.02
    w2 = jax.random.normal(k2, (3, 3, planes // g, planes), dtype) * 0.02
    w3 = jax.random.normal(k3, (1, 1, planes // g, c_in), dtype) * 0.02
    return x, w1, w2, w3


def _conv(x, w, pad, dil):
    return lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=[(pad, pad), (pad, pad)],
        rhs_dilation=(dil, dil),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=CONFIG['group'],
        precision=lax.Precision.HIGHEST)


def _instance_norm(x):
    # affine=False, no running stats: per-(n, c) spatial stats, biased variance.
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps'])


def workload(x, w1, w2, w3):
    dil = CONFIG['dilated']
    h = jnp.transpose(x, (0, 2, 3, 1))
    residual = h
    out = jax.nn.relu(_instance_norm(_conv(h, w1, 0, 1)))
    out = jax.nn.relu(_instance_norm(_conv(out, w2, dil, dil)))
    out = _instance_norm(_conv(out, w3, 0, 1))
    out = jax.nn.relu(out + residual)
    return jnp.transpose(out, (0, 3, 1, 2))
