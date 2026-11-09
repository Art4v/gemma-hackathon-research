"""DeResNetBlockGroupNorm: ConvTranspose3x3 -> GroupNorm -> ReLU -> ConvTranspose3x3 -> GroupNorm -> +x -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DeResNetBlockGroupNorm_2961',
    'batch': 8,
    'inplanes': 128,
    'planes': 128,
    'height': 128,
    'width': 128,
    'num_groups': 1,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'output_padding': 0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    kx, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['inplanes'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kh, kw)
    w1 = jax.random.normal(k1, (c['inplanes'], c['planes'], k, k), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['planes'], c['planes'], k, k), dtype) * 0.02
    g1 = jnp.ones((c['planes'],), dtype)
    b1 = jnp.zeros((c['planes'],), dtype)
    g2 = jnp.ones((c['planes'],), dtype)
    b2 = jnp.zeros((c['planes'],), dtype)
    return x, w1, w2, g1, b1, g2, b2


def _deconv(x, w, stride, padding, output_padding):
    # Transposed conv == dilated-input conv with flipped kernel and swapped in/out.
    k = w.shape[2]
    w_hwio = jnp.transpose(jnp.flip(w, (2, 3)), (2, 3, 0, 1))
    lo = k - 1 - padding
    return lax.conv_general_dilated(
        x, w_hwio,
        window_strides=(1, 1),
        padding=((lo, lo + output_padding), (lo, lo + output_padding)),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )


def _group_norm(x, gamma, beta, groups, eps):
    n, h, w, c = x.shape
    xg = x.reshape(n, h, w, groups, c // groups)
    mean = jnp.mean(xg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(xg - mean), axis=(1, 2, 4), keepdims=True)
    xn = ((xg - mean) * lax.rsqrt(var + eps)).reshape(n, h, w, c)
    return xn * gamma + beta


def workload(x, w1, w2, g1, b1, g2, b2):
    c = CONFIG
    # ponytail: stride=1 and inplanes==planes, so downsample is None; add 1x1 deconv+GN if config changes.
    xh = jnp.transpose(x, (0, 2, 3, 1))
    out = _deconv(xh, w1, c['stride'], c['padding'], c['output_padding'])
    out = jax.nn.relu(_group_norm(out, g1, b1, c['num_groups'], c['eps']))
    out = _deconv(out, w2, 1, c['padding'], 0)
    out = _group_norm(out, g2, b2, c['num_groups'], c['eps'])
    out = jax.nn.relu(out + xh)
    return jnp.transpose(out, (0, 3, 1, 2))
