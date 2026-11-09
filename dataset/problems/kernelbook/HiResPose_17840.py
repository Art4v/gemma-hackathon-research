"""[bf16 variant] HiResPose: 3x (Conv3d + ReLU, maxpool between) -> flatten -> Linear+log_softmax (pose) and Linear (affinity)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'HiResPose_17840',
    'batch': 8,
    'in_channels': 4,
    'depth': 128,
    'height': 128,
    'width': 128,
    'channels': (32, 64, 128),
    'kernel_size': 3,
    'padding': 1,
    'pool': 2,
    'pose_out': 2,
    'affinity_out': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    chans = (c['in_channels'],) + tuple(c['channels'])
    conv_params = []
    for i in range(3):
        w = jax.random.normal(keys[1 + 2 * i], (chans[i + 1], chans[i], k, k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        conv_params += [w, b]
    p = c['pool'] ** 2
    feat = (c['depth'] // p) * (c['height'] // p) * (c['width'] // p) * c['channels'][-1]
    w_pose = jax.random.normal(keys[7], (feat, c['pose_out']), dtype) * 0.02
    b_pose = jax.random.normal(keys[8], (c['pose_out'],), dtype) * 0.02
    w_aff = jax.random.normal(keys[9], (feat, c['affinity_out']), dtype) * 0.02
    b_aff = jax.random.normal(keys[10], (c['affinity_out'],), dtype) * 0.02
    return (x, *conv_params, w_pose, b_pose, w_aff, b_aff)


def _conv_relu(h, w, b):
    pad = CONFIG['padding']
    y = lax.conv_general_dilated(
        h, w, window_strides=(1, 1, 1), padding=[(pad, pad)] * 3,
        dimension_numbers=('NDHWC', 'OIDHW', 'NDHWC'),
        precision=lax.Precision.HIGHEST)
    return jax.nn.relu(y + b)


def _maxpool(h):
    p = CONFIG['pool']
    return lax.reduce_window(h, -jnp.inf, lax.max, (1, p, p, p, 1), (1, p, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, w3, b3, w_pose, b_pose, w_aff, b_aff):
    h = jnp.transpose(x, (0, 2, 3, 4, 1))
    h = _conv_relu(h, w1, b1)
    h = _maxpool(h)
    h = _conv_relu(h, w2, b2)
    h = _maxpool(h)
    h = _conv_relu(h, w3, b3)
    # Flatten in PyTorch NCDHW order.
    h = jnp.transpose(h, (0, 4, 1, 2, 3)).reshape(h.shape[0], -1)
    pose_raw = jnp.dot(h, w_pose, precision=lax.Precision.HIGHEST) + b_pose
    pose_log = jax.nn.log_softmax(pose_raw, axis=1)
    affinity = jnp.dot(h, w_aff, precision=lax.Precision.HIGHEST) + b_aff
    return pose_log, jnp.squeeze(affinity, -1)
