"""[bf16 variant] ChannelSELayer3D: AdaptiveAvgPool3d(1) -> Linear -> ReLU -> Linear -> Sigmoid -> channel-wise scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelSELayer3D_13789',
    'batch': 8,
    'num_channels': 128,
    'reduction_ratio': 2,
    'depth': 16,
    'height': 32,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ch = c['num_channels']
    cr = ch // c['reduction_ratio']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], ch, c['depth'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k1, (ch, cr), dtype) * 0.02
    b1 = jax.random.normal(kb1, (cr,), dtype) * 0.02
    w2 = jax.random.normal(k2, (cr, ch), dtype) * 0.02
    b2 = jax.random.normal(kb2, (ch,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    squeeze = jnp.mean(x, axis=(2, 3, 4))  # (B, C)
    h = jax.nn.relu(squeeze @ w1 + b1)
    gate = jax.nn.sigmoid(h @ w2 + b2)
    return x * gate[:, :, None, None, None]
