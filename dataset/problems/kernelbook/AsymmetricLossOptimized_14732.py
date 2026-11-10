"""[bf16 variant] AsymmetricLossOptimized: sigmoid -> clipped asymmetric log-likelihood -> asymmetric focal weight (pow) -> negated sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AsymmetricLossOptimized_14732',
    'batch': 8192,
    'num_classes': 4096,
    'gamma_neg': 4.0,
    'gamma_pos': 1.0,
    'clip': 0.05,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_classes'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    gamma_neg = CONFIG['gamma_neg']
    gamma_pos = CONFIG['gamma_pos']
    clip = CONFIG['clip']
    eps = CONFIG['eps']

    anti_y = 1.0 - y
    xs_pos = jax.nn.sigmoid(x)
    xs_neg = 1.0 - xs_pos
    if clip is not None and clip > 0:
        xs_neg = jnp.minimum(xs_neg + clip, 1.0)

    loss = y * jnp.log(jnp.maximum(xs_pos, eps))
    loss = loss + anti_y * jnp.log(jnp.maximum(xs_neg, eps))

    if gamma_neg > 0 or gamma_pos > 0:
        base = 1.0 - xs_pos * y - xs_neg * anti_y
        expo = gamma_pos * y + gamma_neg * anti_y
        loss = loss * jnp.power(base, expo)

    return -jnp.sum(loss)
