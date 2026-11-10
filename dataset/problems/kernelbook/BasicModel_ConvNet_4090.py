"""Conv3x3 -> ReLU -> MaxPool2 -> Conv3x3 -> ReLU -> MaxPool2 -> view(-1,4) -> Linear -> ReLU -> Linear -> Softmax."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BasicModel_ConvNet_4090',
    'batch': 32,
    'in_channels': 1,
    'height': 1024,
    'width': 1024,
    'conv1_out': 2,
    'conv2_out': 4,
    'kernel': 3,
    'pool': 2,
    'fc_in': 4,
    'fc_hidden': 8,
    'fc_out': 10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel']
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    # The module pins both fc weights to torch.ones.
    fw1 = jnp.ones((c['fc_in'], c['fc_hidden']), dtype)
    kb3, kb4 = jax.random.split(keys[5])
    fb1 = jax.random.normal(kb3, (c['fc_hidden'],), dtype) * 0.02
    fw2 = jnp.ones((c['fc_hidden'], c['fc_out']), dtype)
    fb2 = jax.random.normal(kb4, (c['fc_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2, fw1, fb1, fw2, fb2


def _conv_relu(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return jax.nn.relu(y + b)


def _maxpool(x):
    p = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _maxpool(_conv_relu(h, w1, b1))
    h = _maxpool(_conv_relu(h, w2, b2))
    # torch view(-1, 4) is a raw reshape of contiguous NCHW memory.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['fc_in'])
    h = jax.nn.relu(jnp.matmul(h, fw1, precision=lax.Precision.HIGHEST) + fb1)
    h = jnp.matmul(h, fw2, precision=lax.Precision.HIGHEST) + fb2
    return jax.nn.softmax(h, axis=1)
