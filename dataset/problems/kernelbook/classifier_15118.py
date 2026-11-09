"""[bf16 variant] Conv3d+InstanceNorm3d+LeakyReLU x4 -> Conv3d -> global avg pool -> LeakyReLU -> Linear (classifier_15118)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'classifier_15118',
    'batch': 8,
    'in_channels': 1,
    'voxel': 128,
    'ef_dim': 32,
    'z_dim': 256,
    'class_num': 128,
    'kernel': 4,
    'stride': 2,
    'padding': 1,
    'eps': 1e-5,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 13)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['voxel'], c['voxel'], c['voxel']), dtype)
    e, k = c['ef_dim'], c['kernel']
    chans = [c['in_channels'], e, e * 2, e * 4, e * 8, c['z_dim']]
    params = []
    for i in range(5):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, k, chans[i], chans[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    wl = jax.random.normal(keys[11], (c['z_dim'], c['class_num']), dtype) * 0.02
    bl = jax.random.normal(keys[12], (c['class_num'],), dtype) * 0.02
    return (x, *params, wl, bl)


def _conv(x, w, b):
    s, p = CONFIG['stride'], CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(s, s, s), padding=[(p, p)] * 3,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    return y + b


def _instance_norm(x):
    mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    var = jnp.var(x, axis=(1, 2, 3), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps'])


def _lrelu(x):
    return jnp.where(x >= 0, x, x * CONFIG['negative_slope'])


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, wl, bl):
    out = jnp.transpose(x, (0, 2, 3, 4, 1))
    for w, b in ((w1, b1), (w2, b2), (w3, b3), (w4, b4)):
        out = _lrelu(_instance_norm(_conv(out, w, b)))
    out = _conv(out, w5, b5)
    z = jnp.mean(out, axis=(1, 2, 3))
    # PyTorch's inplace leaky_relu mutates z, so the returned z is the activated one.
    z = _lrelu(z)
    logits = z @ wl + bl
    return logits, z
