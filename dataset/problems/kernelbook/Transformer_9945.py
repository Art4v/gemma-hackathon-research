"""[bf16 variant] Fast style-transfer Transformer: reflect-pad conv + InstanceNorm + ReLU downsampling, 5 residual blocks, nearest-upsample convs."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Transformer_9945',
    'batch': 8,
    'in_channels': 3,
    'height': 1024,
    'width': 1024,
    'channels': (32, 64, 128),
    'num_res': 5,
    'big_kernel': 9,
    'small_kernel': 3,
    'down_stride': 2,
    'upsample': 2,
    'eps': 1e-5,
}


def _down_specs():
    c0 = CONFIG['in_channels']
    c1, c2, c3 = CONFIG['channels']
    K, k = CONFIG['big_kernel'], CONFIG['small_kernel']
    return [(K, c0, c1), (k, c1, c2), (k, c2, c3)]


def _up_specs():
    c0 = CONFIG['in_channels']
    c1, c2, c3 = CONFIG['channels']
    K, k = CONFIG['big_kernel'], CONFIG['small_kernel']
    return [(k, c3, c2), (k, c2, c1), (K, c1, c0)]


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    key, kx = jax.random.split(key)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    c3 = CONFIG['channels'][2]
    k = CONFIG['small_kernel']
    conv_specs = _down_specs() + [(k, c3, c3)] * (2 * CONFIG['num_res']) + _up_specs()
    n_normed = len(_down_specs()) + 2 * CONFIG['num_res']
    keys = jax.random.split(key, 2 * len(conv_specs))
    params = []
    for i, (kk, ci, co) in enumerate(conv_specs):
        params.append(jax.random.normal(keys[2 * i], (kk, kk, ci, co), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 * i + 1], (co,), dtype) * 0.02)
        if i < n_normed:
            params.append(jnp.ones((co,), dtype))
            params.append(jnp.zeros((co,), dtype))
    return (x, *params)


def _conv(x, w, b, stride):
    p = w.shape[0] // 2
    x = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
    y = lax.conv_general_dilated(
        x, w, (stride, stride), 'VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _inst_norm(x, g, b):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps']) * g + b


def _upsample(x):
    s = CONFIG['upsample']
    return jnp.repeat(jnp.repeat(x, s, axis=1), s, axis=2)


def workload(x, *params):
    it = iter(params)
    h = jnp.transpose(x, (0, 2, 3, 1))
    strides = (1, CONFIG['down_stride'], CONFIG['down_stride'])
    for s in strides:
        w, b, g, be = next(it), next(it), next(it), next(it)
        h = jax.nn.relu(_inst_norm(_conv(h, w, b, s), g, be))
    for _ in range(CONFIG['num_res']):
        w1, b1, g1, be1 = next(it), next(it), next(it), next(it)
        w2, b2, g2, be2 = next(it), next(it), next(it), next(it)
        r = jax.nn.relu(_inst_norm(_conv(h, w1, b1, 1), g1, be1))
        h = h + _inst_norm(_conv(r, w2, b2, 1), g2, be2)
    for i in range(len(_up_specs())):
        w, b = next(it), next(it)
        if i < 2:
            h = _upsample(h)
        h = jax.nn.relu(_conv(h, w, b, 1))
    return jnp.transpose(h, (0, 3, 1, 2))
