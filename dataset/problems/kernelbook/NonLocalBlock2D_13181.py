"""[bf16 variant] NonLocalBlock2D: 1x1 convs g/theta/phi -> flatten -> theta @ phi affinity -> softmax -> matmul g -> reshape -> 1x1 conv W -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NonLocalBlock2D_13181',
    'batch': 8,
    'in_channels': 512,
    'inter_channels': 256,
    'height': 64,
    'width': 64,
    'kernel_size': 1,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    b = CONFIG['batch']
    c = CONFIG['in_channels']
    k = CONFIG['inter_channels']
    h = CONFIG['height']
    w = CONFIG['width']
    ks = CONFIG['kernel_size']

    x = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    g_w = jax.random.normal(keys[1], (k, c, ks, ks), dtype) * 0.02
    g_b = jax.random.normal(keys[2], (k,), dtype) * 0.02
    theta_w = jax.random.normal(keys[3], (k, c, ks, ks), dtype) * 0.02
    theta_b = jax.random.normal(keys[4], (k,), dtype) * 0.02
    phi_w = jax.random.normal(keys[5], (k, c, ks, ks), dtype) * 0.02
    phi_b = jax.random.normal(keys[6], (k,), dtype) * 0.02
    W_w = jax.random.normal(keys[7], (c, k, ks, ks), dtype) * 0.02
    W_b = jax.random.normal(keys[8], (c,), dtype) * 0.02
    return (x, g_w, g_b, theta_w, theta_b, phi_w, phi_b, W_w, W_b)


def _conv(x, weight, bias):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        weight,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + bias
    return jnp.transpose(y, (0, 3, 1, 2))


def workload(x, g_w, g_b, theta_w, theta_b, phi_w, phi_b, W_w, W_b):
    batch_size, _, height, width = x.shape
    inter_channels = g_w.shape[0]

    g_x = jnp.reshape(_conv(x, g_w, g_b), (batch_size, inter_channels, -1))
    g_x = jnp.transpose(g_x, (0, 2, 1))

    theta_x = jnp.reshape(_conv(x, theta_w, theta_b), (batch_size, inter_channels, -1))
    theta_x = jnp.transpose(theta_x, (0, 2, 1))

    phi_x = jnp.reshape(_conv(x, phi_w, phi_b), (batch_size, inter_channels, -1))

    f = jnp.matmul(theta_x, phi_x)
    f_div_C = jax.nn.softmax(f, axis=1)

    y = jnp.matmul(f_div_C, g_x)
    y = jnp.transpose(y, (0, 2, 1))
    y = jnp.reshape(y, (batch_size, inter_channels, height, width))

    W_y = _conv(y, W_w, W_b)
    return W_y + x
