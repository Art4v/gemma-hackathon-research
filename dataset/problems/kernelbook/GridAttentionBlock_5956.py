"""[bf16 variant] GridAttentionBlock: conv1x1(x) + bilinear_resize(conv1x1(g)) -> relu -> conv1x1(->1) -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GridAttentionBlock_5956',
    'batch': 8,
    'in_channels': 128,
    'inter_channels': 128,
    'gating_channels': 128,
    'height': 128,
    'width': 128,
    'gate_height': 128,
    'gate_width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 8)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    g = jax.random.uniform(keys[1], (c['batch'], c['gating_channels'], c['gate_height'], c['gate_width']), dtype)
    theta_w = jax.random.normal(keys[2], (1, 1, c['in_channels'], c['inter_channels']), dtype) * 0.02
    theta_b = jax.random.normal(keys[3], (c['inter_channels'],), dtype) * 0.02
    phi_w = jax.random.normal(keys[4], (1, 1, c['gating_channels'], c['inter_channels']), dtype) * 0.02
    phi_b = jax.random.normal(keys[5], (c['inter_channels'],), dtype) * 0.02
    psi_w = jax.random.normal(keys[6], (1, 1, c['inter_channels'], 1), dtype) * 0.02
    psi_b = jax.random.normal(keys[7], (1,), dtype) * 0.02
    return x, g, theta_w, theta_b, phi_w, phi_b, psi_w, psi_b


def _conv1x1(x_nhwc, w_hwio, b):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, g, theta_w, theta_b, phi_w, phi_b, psi_w, psi_b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    g_nhwc = jnp.transpose(g, (0, 2, 3, 1))
    theta_x = _conv1x1(x_nhwc, theta_w, theta_b)
    phi_g = _conv1x1(g_nhwc, phi_w, phi_b)
    if phi_g.shape != theta_x.shape:
        # Bilinear, align_corners=False, no antialias: matches F.interpolate(mode='bilinear').
        phi_g = jax.image.resize(phi_g, theta_x.shape, method='linear', antialias=False)
    f = jax.nn.relu(theta_x + phi_g)
    out = jax.nn.sigmoid(_conv1x1(f, psi_w, psi_b))
    return jnp.transpose(out, (0, 3, 1, 2))
