"""DupCNN2: 6x (Conv2d3x3 + ReLU + MaxPool2) -> flatten -> Linear + ReLU -> Linear -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DupCNN2_12006',
    'batch': 128,
    'height': 256,
    'width': 256,
    'channels': (4, 16, 32, 64, 128, 256, 512),
    'kernel_size': 3,
    'padding': 1,
    'pool': 2,
    'final_spatial': 4,
    'fc_hidden': 256,
    'output_size': 4,
}


def create_inputs(dtype=jnp.float32):
    chans = CONFIG['channels']
    k = CONFIG['kernel_size']
    n_conv = len(chans) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_conv + 4)
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], chans[0], CONFIG['height'], CONFIG['width']), dtype)
    params = []
    for i in range(n_conv):
        w = jax.random.normal(keys[1 + 2 * i], (chans[i + 1], chans[i], k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    flat = chans[-1] * CONFIG['final_spatial'] ** 2
    hidden, out = CONFIG['fc_hidden'], CONFIG['output_size']
    kf = keys[1 + 2 * n_conv:]
    w_fc1 = jax.random.normal(kf[0], (flat, hidden), dtype) * 0.02
    b_fc1 = jax.random.normal(kf[1], (hidden,), dtype) * 0.02
    w_fc2 = jax.random.normal(kf[2], (hidden, out), dtype) * 0.02
    b_fc2 = jax.random.normal(kf[3], (out,), dtype) * 0.02
    return (x, *params, w_fc1, b_fc1, w_fc2, b_fc2)


def workload(x, *params):
    conv_params = params[:-4]
    w_fc1, b_fc1, w_fc2, b_fc2 = params[-4:]
    pad = CONFIG['padding']
    p = CONFIG['pool']
    h = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    for i in range(0, len(conv_params), 2):
        w, b = conv_params[i], conv_params[i + 1]
        h = lax.conv_general_dilated(
            h, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
            precision=lax.Precision.HIGHEST)
        h = jax.nn.relu(h + b)
        h = lax.reduce_window(h, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')
    h = jnp.transpose(h, (0, 3, 1, 2))  # back to NCHW so flatten order matches torch .view
    h = h.reshape(-1, w_fc1.shape[0])
    h = jax.nn.relu(jnp.dot(h, w_fc1, precision=lax.Precision.HIGHEST) + b_fc1)
    h = jnp.dot(h, w_fc2, precision=lax.Precision.HIGHEST) + b_fc2
    return jax.nn.sigmoid(h)
