"""[bf16 variant] Binary focal loss: sigmoid -> clamp -> masked focal pos/neg terms -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryFocalLoss_12816',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'alpha': 3.0,
    'gamma': 2.0,
    'smooth': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.normal(k1, shape, dtype)
    # Binary labels so the target==1 / target==0 masks are non-trivial.
    target = jax.random.bernoulli(k2, 0.5, shape).astype(dtype)
    return output, target


def workload(output, target):
    alpha, gamma, smooth = CONFIG['alpha'], CONFIG['gamma'], CONFIG['smooth']
    prob = jnp.clip(jax.nn.sigmoid(output), smooth, 1.0 - smooth)
    pos_mask = (target == 1).astype(output.dtype)
    neg_mask = (target == 0).astype(output.dtype)
    pos_loss = -(pos_mask * jnp.power(1.0 - prob, gamma)) * jnp.log(prob)
    neg_loss = -alpha * (neg_mask * jnp.power(prob, gamma)) * jax.nn.log_sigmoid(-output)
    return jnp.mean(pos_loss + neg_loss)
