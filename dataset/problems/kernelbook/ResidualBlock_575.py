"""Conv3x3 -> InstanceNorm -> ReLU -> Conv3x3 -> InstanceNorm -> residual add -> ReLU (ResidualBlock)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlock_575',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'dilation': 1,
    'groups': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    n, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    k, g = CONFIG['kernel_size'], CONFIG['groups']
    kx, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (n, c, h, w), dtype)
    w1 = jax.random.normal(k1, (c, c // g, k, k), dtype) * 0.02
    w2 = jax.random.normal(k2, (c, c // g, k, k), dtype) * 0.02
    gamma1 = jnp.ones((c,), dtype)
    beta1 = jnp.zeros((c,), dtype)
    gamma2 = jnp.ones((c,), dtype)
    beta2 = jnp.zeros((c,), dtype)
    return x, w1, gamma1, beta1, w2, gamma2, beta2


def _conv(x_nhwc, w_oihw):
    d = CONFIG['dilation']
    w_hwio = jnp.transpose(w_oihw, (2, 3, 1, 0))
    return lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((d, d), (d, d)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=CONFIG['groups'],
        precision=lax.Precision.HIGHEST,
    )


def _instance_norm(y_nhwc, gamma, beta):
    mean = jnp.mean(y_nhwc, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(y_nhwc - mean), axis=(1, 2), keepdims=True)
    return (y_nhwc - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, gamma1, beta1, w2, gamma2, beta2):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = jax.nn.relu(_instance_norm(_conv(xh, w1), gamma1, beta1))
    y = _instance_norm(_conv(y, w2), gamma2, beta2)
    out = jax.nn.relu(xh + y)
    return jnp.transpose(out, (0, 3, 1, 2))
