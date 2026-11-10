"""Per-channel fixed-kernel conv2d (vertical, horizontal) -> gradient magnitude sqrt(v^2 + h^2 + eps) -> concat."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Get_gradient_7834',
    'batch': 8,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'used_channels': 3,
    'padding': 2,
    'strides': (1, 1),
    'eps': 1e-06,
    'kernel_h': [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    'kernel_v': [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight_h = jnp.array(CONFIG['kernel_h'], dtype)[None, None, :, :]
    weight_v = jnp.array(CONFIG['kernel_v'], dtype)[None, None, :, :]
    return x, weight_h, weight_v


def _conv2d(x, weight):
    pad = CONFIG['padding']
    out = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        jnp.transpose(weight, (2, 3, 1, 0)),
        window_strides=CONFIG['strides'],
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(out, (0, 3, 1, 2))


def workload(x, weight_h, weight_v):
    outputs = []
    for i in range(CONFIG['used_channels']):
        channel = x[:, i:i + 1]
        grad_v = _conv2d(channel, weight_v)
        grad_h = _conv2d(channel, weight_h)
        outputs.append(jnp.sqrt(jnp.square(grad_v) + jnp.square(grad_h) + CONFIG['eps']))
    return jnp.concatenate(outputs, axis=1)
