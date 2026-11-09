"""[bf16 variant] Project & Excite 3D: tri-axial avg-pool squeeze -> 1x1x1 conv -> ReLU -> 1x1x1 conv -> sigmoid -> scale input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ProjectExciteLayer_2703',
    'batch': 8,
    'num_channels': 128,
    'reduction_ratio': 2,
    'depth': 16,
    'height': 32,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG['num_channels']
    cr = c // CONFIG['reduction_ratio']
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], c, CONFIG['depth'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(k_w1, (c, cr), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (cr,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (cr, c), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    # x: (B, C, D, H, W); w1: (C, Cr); w2: (Cr, C)
    sq_w = jnp.mean(x, axis=(2, 3), keepdims=True)  # (B, C, 1, 1, W)
    sq_h = jnp.mean(x, axis=(2, 4), keepdims=True)  # (B, C, 1, H, 1)
    sq_d = jnp.mean(x, axis=(3, 4), keepdims=True)  # (B, C, D, 1, 1)
    squeeze = sq_w + sq_h + sq_d  # broadcast to (B, C, D, H, W)
    hi = lax.Precision.HIGHEST
    h = jnp.einsum('bcdhw,cr->brdhw', squeeze, w1, precision=hi)
    h = jax.nn.relu(h + b1[None, :, None, None, None])
    g = jnp.einsum('brdhw,rc->bcdhw', h, w2, precision=hi)
    g = jax.nn.sigmoid(g + b2[None, :, None, None, None])
    return x * g
