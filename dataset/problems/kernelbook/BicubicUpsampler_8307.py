"""[bf16 variant] ReplicationPad2d -> depthwise ConvTranspose2d along W -> depthwise ConvTranspose2d along H (bicubic 2x upsample)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BicubicUpsampler_8307',
    'batch': 32,
    'channels': 1,
    'height': 1024,
    'width': 1024,
    'scale': 2,
    'ksize': 8,  # 2 * scale * 2, +1 if scale is odd
    'pad': 2,
}


def _bicubic_taps(dtype):
    s, k = CONFIG['scale'], CONFIG['ksize']
    coord = (jnp.arange(k, dtype=jnp.float32) + 0.5) / s - k * 0.5 / s
    a = jnp.abs(coord)
    a2, a3 = a ** 2, a ** 3
    outer = (-0.5 * a3 + 2.5 * a2 - 4.0 * a + 2.0) * ((a > 1.0) & (a < 2.0))
    inner = (1.5 * a3 - 2.5 * a2 + 1.0) * (a <= 1.0)
    return (outer + inner).astype(dtype)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    taps = jnp.broadcast_to(_bicubic_taps(dtype), (CONFIG['channels'], CONFIG['ksize']))
    return x, taps, taps


def _transposed_depthwise(h, w, axis):
    # ConvTranspose == dilated-input conv with flipped kernel, pad K-1-p per side.
    s = CONFIG['scale']
    c, k = w.shape
    edge = k - 1 - (2 * s * 2 - s // 2)
    kern = jnp.flip(w, -1).T  # (K, C)
    if axis == 'w':
        kern = kern.reshape(1, k, 1, c)
        padding, dil = ((0, 0), (edge, edge)), (1, s)
    else:
        kern = kern.reshape(k, 1, 1, c)
        padding, dil = ((edge, edge), (0, 0)), (s, 1)
    return lax.conv_general_dilated(
        h, kern, window_strides=(1, 1), padding=padding, lhs_dilation=dil,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=c,
        precision=lax.Precision.HIGHEST)


def workload(x, wx, wy):
    p = CONFIG['pad']
    xp = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='edge')
    h = jnp.transpose(xp, (0, 2, 3, 1))
    h = _transposed_depthwise(h, wx, 'w')
    h = _transposed_depthwise(h, wy, 'h')
    return jnp.transpose(h, (0, 3, 1, 2))
