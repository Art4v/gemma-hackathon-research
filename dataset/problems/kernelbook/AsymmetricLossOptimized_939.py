"""[bf16 variant] Asymmetric multi-label loss: sigmoid, clipped negative probs, clamped logs, asymmetric focal power weighting, negated sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AsymmetricLossOptimized_939',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'gamma_neg': 4,
    'gamma_pos': 1,
    'clip': 0.05,
    'eps': 1e-08,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_x, shape, dtype)
    y = jax.random.uniform(key_y, shape, dtype)
    return x, y


def workload(x, y):
    gamma_neg = CONFIG['gamma_neg']
    gamma_pos = CONFIG['gamma_pos']
    clip = CONFIG['clip']
    eps = CONFIG['eps']

    targets = y
    anti_targets = 1 - y
    xs_pos = jax.nn.sigmoid(x)
    xs_neg = 1.0 - xs_pos
    if clip is not None and clip > 0:
        xs_neg = jnp.minimum(xs_neg + clip, 1.0)

    loss = targets * jnp.log(jnp.maximum(xs_pos, eps))
    loss = loss + anti_targets * jnp.log(jnp.maximum(xs_neg, eps))

    if gamma_neg > 0 or gamma_pos > 0:
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = jnp.power(
            1 - xs_pos - xs_neg,
            gamma_pos * targets + gamma_neg * anti_targets,
        )
        loss = loss * asymmetric_w

    return -jnp.sum(loss)
