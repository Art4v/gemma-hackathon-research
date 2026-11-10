"""[bf16 variant] NetRes: conv3x3+relu+maxpool x2, conv3x3+relu+residual+maxpool, view(-1,256), fc+relu, fc."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NetRes_3452',
    'batch': 2048,
    'in_chans': 3,
    'height': 64,
    'width': 64,
    'n_chans1': 32,
    'kernel_size': 3,
    'padding': 1,
    'pool': 2,
    'fc_hidden': 32,
    'fc_out': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 11)
    n1, n2, ks = c['n_chans1'], c['n_chans1'] // 2, c['kernel_size']
    fc_in = 4 * 4 * n1 // 2
    x = jax.random.uniform(k[0], (c['batch'], c['in_chans'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k[1], (n1, c['in_chans'], ks, ks), dtype) * 0.02
    b1 = jax.random.normal(k[2], (n1,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (n2, n1, ks, ks), dtype) * 0.02
    b2 = jax.random.normal(k[4], (n2,), dtype) * 0.02
    w3 = jax.random.normal(k[5], (n2, n2, ks, ks), dtype) * 0.02
    b3 = jax.random.normal(k[6], (n2,), dtype) * 0.02
    fw1 = jax.random.normal(k[7], (fc_in, c['fc_hidden']), dtype) * 0.02
    fb1 = jax.random.normal(k[8], (c['fc_hidden'],), dtype) * 0.02
    fw2 = jax.random.normal(k[9], (c['fc_hidden'], c['fc_out']), dtype) * 0.02
    fb2 = jax.random.normal(k[10], (c['fc_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return y + b


def _pool(x):
    p = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, w3, b3, fw1, fb1, fw2, fb2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _pool(jax.nn.relu(_conv(h, w1, b1)))
    h = _pool(jax.nn.relu(_conv(h, w2, b2)))
    h = _pool(jax.nn.relu(_conv(h, w3, b3)) + h)
    # back to NCHW so the flatten matches PyTorch's .view element order
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, fw1.shape[0])
    h = jax.nn.relu(h @ fw1 + fb1)
    return h @ fw2 + fb2
