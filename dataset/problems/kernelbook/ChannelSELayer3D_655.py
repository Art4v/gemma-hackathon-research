"""ChannelSELayer3D: global avg pool -> Linear -> ReLU -> Linear -> sigmoid -> channel-wise scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelSELayer3D_655',
    'batch': 8,
    'num_channels': 128,
    'reduction_ratio': 2,
    'depth': 32,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    c = CONFIG['num_channels']
    cr = c // CONFIG['reduction_ratio']
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        k_x, (b, c, CONFIG['depth'], CONFIG['height'], CONFIG['width']), dtype)
    fc1_w = jax.random.normal(k_w1, (c, cr), dtype) * 0.02
    fc1_b = jax.random.normal(k_b1, (cr,), dtype) * 0.02
    fc2_w = jax.random.normal(k_w2, (cr, c), dtype) * 0.02
    fc2_b = jax.random.normal(k_b2, (c,), dtype) * 0.02
    return x, fc1_w, fc1_b, fc2_w, fc2_b


def workload(x, fc1_w, fc1_b, fc2_w, fc2_b):
    # x: (B, C, D, H, W); weights are (in, out).
    squeeze = jnp.mean(x, axis=(2, 3, 4))
    h = jax.nn.relu(squeeze @ fc1_w + fc1_b)
    gate = jax.nn.sigmoid(h @ fc2_w + fc2_b)
    return x * gate[:, :, None, None, None]
