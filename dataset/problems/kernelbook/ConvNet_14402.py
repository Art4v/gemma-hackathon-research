"""LeNet++: 6x (Conv2d 5x5 + PReLU) with 2x2 maxpool every 2 convs, flatten, Linear + PReLU, Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvNet_14402',
    'batch': 24576,
    'in_channels': 1,
    'height': 24,
    'width': 24,
    'conv_channels': (32, 32, 64, 64, 128, 128),
    'kernel_size': 5,
    'padding': 2,
    'pool': 2,
    'fc1_out': 2,
    'num_classes': 4,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.float32):
    chans = CONFIG['conv_channels']
    k = CONFIG['kernel_size']
    n_keys = 1 + 2 * len(chans) + 4
    keys = jax.random.split(jax.random.key(0), n_keys)
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    params = []
    cin = CONFIG['in_channels']
    ki = 1
    for cout in chans:
        w = jax.random.normal(keys[ki], (cout, cin, k, k), dtype) * 0.02
        b = jax.random.normal(keys[ki + 1], (cout,), dtype) * 0.02
        a = jnp.full((1,), CONFIG['prelu_init'], dtype)
        params += [w, b, a]
        cin = cout
        ki += 2
    pooled = CONFIG['height'] // CONFIG['pool'] ** 3
    flat = chans[-1] * pooled * pooled
    fc1_w = jax.random.normal(keys[ki], (flat, CONFIG['fc1_out']), dtype) * 0.02
    fc1_b = jax.random.normal(keys[ki + 1], (CONFIG['fc1_out'],), dtype) * 0.02
    fc1_a = jnp.full((1,), CONFIG['prelu_init'], dtype)
    fc2_w = jax.random.normal(keys[ki + 2], (CONFIG['fc1_out'], CONFIG['num_classes']), dtype) * 0.02
    fc2_b = jax.random.normal(keys[ki + 3], (CONFIG['num_classes'],), dtype) * 0.02
    return (x, *params, fc1_w, fc1_b, fc1_a, fc2_w, fc2_b)


def _prelu(x, a):
    return jnp.where(x >= 0, x, a[0] * x)


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _maxpool(x):
    s = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, s, s, 1), (1, s, s, 1), 'VALID')


def workload(x, *params):
    n_conv = len(CONFIG['conv_channels'])
    conv_params = params[:3 * n_conv]
    fc1_w, fc1_b, fc1_a, fc2_w, fc2_b = params[3 * n_conv:]
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i in range(n_conv):
        w, b, a = conv_params[3 * i:3 * i + 3]
        h = _prelu(_conv(h, w, b), a)
        if i % 2 == 1:
            h = _maxpool(h)
    # Flatten in NCHW order to match torch's view(-1, C*H*W).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    feat = _prelu(jnp.dot(h, fc1_w, precision=lax.Precision.HIGHEST) + fc1_b, fc1_a)
    logits = jnp.dot(feat, fc2_w, precision=lax.Precision.HIGHEST) + fc2_b
    return feat, logits
