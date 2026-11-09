"""[bf16 variant] LeNet++: 6x (conv5x5 + PReLU) with maxpool2 every 2 convs, Linear + PReLU, Linear, log_softmax."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LeNetPP_16004',
    'batch': 32768,
    'in_channels': 1,
    'height': 24,
    'width': 24,
    'conv_channels': (32, 32, 64, 64, 128, 128),
    'kernel_size': 5,
    'padding': 2,
    'pool': 2,
    'dim_hidden': 2,
    'num_classes': 10,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 16)
    b, c, h, w = CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']
    k = CONFIG['kernel_size']
    x = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    conv_params = []
    cin = c
    for i, cout in enumerate(CONFIG['conv_channels']):
        conv_params.append(jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02)
        conv_params.append(jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02)
        cin = cout
    n_pools = len(CONFIG['conv_channels']) // 2
    flat = cin * (h // CONFIG['pool'] ** n_pools) * (w // CONFIG['pool'] ** n_pools)
    dh, nc = CONFIG['dim_hidden'], CONFIG['num_classes']
    ip1_w = jax.random.normal(keys[13], (flat, dh), dtype) * 0.02
    ip1_b = jax.random.normal(keys[14], (dh,), dtype) * 0.02
    ip2_w = jax.random.normal(keys[15], (dh, nc), dtype) * 0.02
    ip2_b = jnp.zeros((nc,), dtype)
    # One shared alpha per nn.PReLU(): 6 conv PReLUs then prelu_ip1.
    alphas = jnp.full((len(CONFIG['conv_channels']) + 1,), CONFIG['prelu_init'], dtype)
    return (x, *conv_params, ip1_w, ip1_b, ip2_w, ip2_b, alphas)


def _prelu(x, a):
    return jnp.where(x >= 0, x, a * x)


def _maxpool(x):
    p = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, *params):
    n_conv = len(CONFIG['conv_channels'])
    conv_params = params[:2 * n_conv]
    ip1_w, ip1_b, ip2_w, ip2_b, alphas = params[2 * n_conv:]
    pad = CONFIG['padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i in range(n_conv):
        w, b = conv_params[2 * i], conv_params[2 * i + 1]
        h = lax.conv_general_dilated(
            h, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            precision=lax.Precision.HIGHEST,
        ) + b
        h = _prelu(h, alphas[i])
        if i % 2 == 1:
            h = _maxpool(h)
    # Flatten in PyTorch (C, H, W) order.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    ip1 = _prelu(jnp.dot(h, ip1_w, precision=lax.Precision.HIGHEST) + ip1_b, alphas[n_conv])
    ip2 = jnp.dot(ip1, ip2_w, precision=lax.Precision.HIGHEST) + ip2_b
    return ip1, jax.nn.log_softmax(ip2, axis=1)
