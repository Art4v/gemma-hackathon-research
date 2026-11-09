"""[bf16 variant] Channelwise attention: global avg pool -> Linear -> ReLU -> Linear -> sigmoid -> broadcast to input shape, plus mean of gates."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelwiseAttention_3495',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'reduction': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    hidden = c // CONFIG['reduction']
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    w1 = jax.random.normal(k_w1, (c, hidden), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (hidden,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (hidden, c), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    feats = jnp.mean(x, axis=(2, 3))
    feats = jax.nn.relu(feats @ w1 + b1)
    feats = jax.nn.sigmoid(feats @ w2 + b2)
    ca_act_reg = jnp.mean(feats)
    out = jnp.broadcast_to(feats[:, :, None, None], x.shape)
    return out, ca_act_reg
