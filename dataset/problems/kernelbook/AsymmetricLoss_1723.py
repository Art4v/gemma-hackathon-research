"""[bf16 variant] AsymmetricLoss: sigmoid -> negative-probability clip -> clamped log BCE terms -> asymmetric focal weight -> negated sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AsymmetricLoss_1723',
    'batch': 8,
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
    key = jax.random.key(0)
    k_x, k_y = jax.random.split(key)
    x = jax.random.uniform(k_x, shape, dtype)
    y = jax.random.uniform(k_y, shape, dtype)
    return x, y


def workload(x, y):
    gamma_neg = CONFIG['gamma_neg']
    gamma_pos = CONFIG['gamma_pos']
    clip = CONFIG['clip']
    eps = CONFIG['eps']

    x_sigmoid = jax.nn.sigmoid(x)
    xs_pos = x_sigmoid
    xs_neg = 1 - x_sigmoid
    if clip is not None and clip > 0:
        xs_neg = jnp.minimum(xs_neg + clip, 1)

    los_pos = y * jnp.log(jnp.clip(xs_pos, eps, 1 - eps))
    los_neg = (1 - y) * jnp.log(jnp.clip(xs_neg, eps, 1 - eps))
    loss = los_pos + los_neg

    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
        one_sided_w = jnp.power(1 - pt, one_sided_gamma)
        loss = loss * one_sided_w

    return -jnp.sum(loss)
