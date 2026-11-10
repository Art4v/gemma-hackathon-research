"""DownBlock: Conv+InstanceNorm+PReLU -> ConvTranspose+LeakyReLU -> 1x1 modulated residual -> Conv+InstanceNorm+PReLU -> 1x1 modulated residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DownBlock_8282',
    'batch': 8,
    'input_size': 128,
    'output_size': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 4,
    'eps': 1e-5,
    'prelu_init': 0.25,
    'leaky_slope': 0.01,
}

_DN = ('NHWC', 'HWIO', 'NHWC')


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    cin, cout, k = c['input_size'], c['output_size'], c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(keys[0], (c['batch'], cin, c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (cout, cin, k, k), dtype) * 0.02        # Conv2d OIHW
    w2 = jax.random.normal(keys[2], (cout, cout, k, k), dtype) * 0.02       # ConvTranspose2d IOHW
    w3 = jax.random.normal(keys[3], (cout, cout, k, k), dtype) * 0.02       # Conv2d OIHW
    lw1 = jax.random.normal(keys[4], (2 * cout, cin, 1, 1), dtype) * 0.02
    lw2 = jax.random.normal(keys[5], (2 * cout, cout, 1, 1), dtype) * 0.02
    a1 = jnp.full((1,), c['prelu_init'], dtype)
    a3 = jnp.full((1,), c['prelu_init'], dtype)
    return x, w1, w2, w3, lw1, lw2, a1, a3


def _conv(x, w_oihw, stride, pad):
    return lax.conv_general_dilated(
        x, jnp.transpose(w_oihw, (2, 3, 1, 0)), (stride, stride),
        [(pad, pad), (pad, pad)], dimension_numbers=_DN)


def _deconv(x, w_iohw, stride, pad):
    # Transposed conv = dilate input by stride, pad by k-1-p (negative here, i.e. crop), VALID conv with flipped kernel.
    k = w_iohw.shape[-1]
    q = k - 1 - pad
    xp = lax.pad(x, jnp.zeros((), x.dtype),
                 [(0, 0, 0), (q, q, stride - 1), (q, q, stride - 1), (0, 0, 0)])
    w = jnp.transpose(w_iohw[:, :, ::-1, ::-1], (2, 3, 0, 1))  # (kh, kw, cin, cout)
    return lax.conv_general_dilated(xp, w, (1, 1), 'VALID', dimension_numbers=_DN)


def _instance_norm(x, eps):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + eps)


def _prelu(x, a):
    return jnp.where(x >= 0, x, a[0] * x)


def _conv_block(x, w, a, stride, pad, eps):
    return _prelu(_instance_norm(_conv(x, w, stride, pad), eps), a)


def _local(x, w):
    out = jnp.einsum('nhwc,oc->nhwo', x, w[:, :, 0, 0])
    return jnp.split(out, 2, axis=-1)


def workload(x, w1, w2, w3, lw1, lw2, a1, a3):
    c = CONFIG
    s, p, eps = c['stride'], c['padding'], c['eps']
    x = jnp.transpose(x, (0, 2, 3, 1))
    lr = _conv_block(x, w1, a1, s, p, eps)
    hr = _deconv(lr, w2, s, p)
    hr = jnp.where(hr >= 0, hr, c['leaky_slope'] * hr)
    mean, var = _local(x, lw1)
    residue = mean + hr * (1 + var)
    l_residue = _conv_block(residue, w3, a3, s, p, eps)
    mean, var = _local(lr, lw2)
    out = mean + l_residue * (1 + var)
    return jnp.transpose(out, (0, 3, 1, 2))
