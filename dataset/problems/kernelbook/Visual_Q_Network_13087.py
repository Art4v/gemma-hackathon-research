"""[bf16 variant] Visual Q-network: conv8x8s4p2+ReLU -> conv4x4s2+ReLU -> flatten -> Linear+ReLU -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Visual_Q_Network_13087',
    'batch': 2048,
    'num_frame': 4,
    'height': 81,
    'width': 81,
    'c1_out': 16,
    'c1_kernel': 8,
    'c1_stride': 4,
    'c1_pad': 2,
    'c2_out': 32,
    'c2_kernel': 4,
    'c2_stride': 2,
    'fc1_in': 32 * 81,
    'fc1_out': 256,
    'num_action': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 9)
    image = jax.random.uniform(ks[0], (c['batch'], c['num_frame'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(ks[1], (c['c1_kernel'], c['c1_kernel'], c['num_frame'], c['c1_out']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['c1_out'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['c2_kernel'], c['c2_kernel'], c['c1_out'], c['c2_out']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['c2_out'],), dtype) * 0.02
    fw1 = jax.random.normal(ks[5], (c['fc1_in'], c['fc1_out']), dtype) * 0.02
    fb1 = jax.random.normal(ks[6], (c['fc1_out'],), dtype) * 0.02
    fw2 = jax.random.normal(ks[7], (c['fc1_out'], c['num_action']), dtype) * 0.02
    fb2 = jax.random.normal(ks[8], (c['num_action'],), dtype) * 0.02
    return image, w1, b1, w2, b2, fw1, fb1, fw2, fb2


def _conv(x, w, b, stride, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(image, w1, b1, w2, b2, fw1, fb1, fw2, fb2):
    c = CONFIG
    x = jnp.transpose(image, (0, 2, 3, 1))
    x = jax.nn.relu(_conv(x, w1, b1, c['c1_stride'], c['c1_pad']))
    x = jax.nn.relu(_conv(x, w2, b2, c['c2_stride'], 0))
    # Flatten in NCHW (C, H, W) order to match torch's view(-1, 32*81).
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape(-1, fw1.shape[0])
    x = jax.nn.relu(x @ fw1 + fb1)
    return x @ fw2 + fb2
