"""[m1 M1_batch variant of kb/l2/19_ConvTranspose2d_GELU_GroupNorm] ConvTranspose2d -> GELU (exact) -> GroupNorm with shared weights, applied over a new leading axis of 4 views with H and W shrunk from 256 to 128."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '19_ConvTranspose2d_GELU_GroupNorm_m1_batch',
    'num_views': 4,
    'batch': 8,
    'in_channels': 64,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'num_groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k0, (c['num_views'], c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k1, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k2, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def _conv_gelu_groupnorm(x, w, b, gamma, beta):
    s = CONFIG['stride']
    K = w.shape[2]
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    # Transposed conv as dilated-input conv with flipped kernel.
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))  # (kH, kW, Cin, Cout)
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=[(K - 1, K - 1), (K - 1, K - 1)],
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = jax.nn.gelu(y, approximate=False)
    N, H, W, C = y.shape
    yg = y.reshape(N, H, W, G, C // G)
    mean = jnp.mean(yg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 4), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + eps)
    y = yg.reshape(N, H, W, C) * gamma + beta
    return jnp.transpose(y, (0, 3, 1, 2))


def workload(x, w, b, gamma, beta):
    return jax.vmap(_conv_gelu_groupnorm, in_axes=(0, None, None, None, None))(x, w, b, gamma, beta)
