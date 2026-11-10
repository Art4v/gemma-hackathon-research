"""[bf16 variant] DQN: reshape -> conv8s4p1+ReLU -> conv4s2+ReLU -> conv3+ReLU -> flatten -> fc+ReLU -> fc."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DQN_7571',
    'batch': 1024,
    'in_channels': 1,
    'height': 185,
    'width': 95,
    'c1': 32, 'k1': 8, 's1': 4, 'p1': 1,
    'c2': 64, 'k2': 4, 's2': 2, 'p2': 0,
    'c3': 128, 'k3': 3, 's3': 1, 'p3': 0,
    'flat_features': 128 * 19 * 8,
    'fc1_units': 512,
    'action_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 11)
    state = jax.random.uniform(ks[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(ks[1], (c['c1'], c['in_channels'], c['k1'], c['k1']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['c1'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['c2'], c['c1'], c['k2'], c['k2']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['c2'],), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (c['c3'], c['c2'], c['k3'], c['k3']), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (c['c3'],), dtype) * 0.02
    fw1 = jax.random.normal(ks[7], (c['flat_features'], c['fc1_units']), dtype) * 0.02
    fb1 = jax.random.normal(ks[8], (c['fc1_units'],), dtype) * 0.02
    fw2 = jax.random.normal(ks[9], (c['fc1_units'], c['action_size']), dtype) * 0.02
    fb2 = jax.random.normal(ks[10], (c['action_size'],), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2


def _conv_relu(x, w, b, stride, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(state, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2):
    c = CONFIG
    x = state.reshape(-1, c['in_channels'], c['height'], c['width'])
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = _conv_relu(x, w1, b1, c['s1'], c['p1'])
    x = _conv_relu(x, w2, b2, c['s2'], c['p2'])
    x = _conv_relu(x, w3, b3, c['s3'], c['p3'])
    # Flatten in PyTorch's (C, H, W) order so fc1 rows line up.
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape(-1, fw1.shape[0])
    x = jax.nn.relu(x @ fw1 + fb1)
    return x @ fw2 + fb2
