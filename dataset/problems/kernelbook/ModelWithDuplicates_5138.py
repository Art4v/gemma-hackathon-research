"""Conv2d(k5) -> ReLU -> Tanh -> Conv2d(k3) -> ReLU -> Tanh (shared activation list)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ModelWithDuplicates_5138',
    'batch': 32,
    'in_channels': 3,
    'height': 512,
    'width': 512,
    'conv1_out': 10,
    'conv1_kernel': 5,
    'conv2_out': 20,
    'conv2_kernel': 3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k_w1, (c['conv1_kernel'], c['conv1_kernel'], c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['conv2_kernel'], c['conv2_kernel'], c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['conv2_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x_nhwc, w_hwio, b):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jnp.tanh(jax.nn.relu(_conv(h, w1, b1)))
    # post_conv2 is the same ModuleList as post_conv1: relu then tanh again
    h = jnp.tanh(jax.nn.relu(_conv(h, w2, b2)))
    return jnp.transpose(h, (0, 3, 1, 2))
