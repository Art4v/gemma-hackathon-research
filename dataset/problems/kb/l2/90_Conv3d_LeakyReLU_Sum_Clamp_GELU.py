"""Conv3d -> LeakyReLU(0.2) -> add per-channel tensor -> clamp[-1,1] -> GELU (exact)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '90_Conv3d_LeakyReLU_Sum_Clamp_GELU',
    'batch_size': 32,
    'in_channels': 8,
    'out_channels': 64,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'negative_slope': 0.2,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b, k_s = jax.random.split(jax.random.key(0), 4)
    k = c['kernel_size']
    x = jax.random.uniform(
        k_x, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (k, k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    sum_tensor = jax.random.normal(k_s, (c['out_channels'], 1, 1, 1), dtype)
    return x, w, b, sum_tensor


def workload(x, w, b, sum_tensor):
    c = CONFIG
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    y = lax.conv_general_dilated(
        x_ndhwc, w,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b
    y = jnp.transpose(y, (0, 4, 1, 2, 3))
    y = jnp.where(y >= 0, y, y * c['negative_slope'])
    y = y + sum_tensor
    y = jnp.clip(y, c['clamp_min'], c['clamp_max'])
    y = jax.nn.gelu(y, approximate=False)
    return y
