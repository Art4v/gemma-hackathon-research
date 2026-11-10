"""RingLoss_15456: soft-target cross entropy over axis 1 plus auto-weighted ring loss on the axis-1 L2 norm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RingLoss_15456',
    'batch': 16,
    'num_classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'radius_size': 1,
    'radius_init': -1.0,
    'loss_weight': 1.0,
    'softmax_loss_weight': 1.0,
    'mean_clamp_min': 0.5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    radius = jnp.full((CONFIG['radius_size'],), CONFIG['radius_init'], dtype)
    return x, y, radius


def workload(x, y, radius):
    axis = CONFIG['class_axis']
    log_probs = jax.nn.log_softmax(x, axis=axis)
    cross_entropy = jnp.mean(-jnp.sum(y * log_probs, axis=axis))
    softmax = cross_entropy * CONFIG['softmax_loss_weight']

    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))
    norm_mean = jnp.mean(norm)
    r = jnp.where(radius[0] < 0, norm_mean, radius[0])

    diff = (norm - r) / jnp.maximum(norm_mean, CONFIG['mean_clamp_min'])
    ringloss = jnp.mean(jnp.square(jnp.abs(diff))) * CONFIG['loss_weight']
    return softmax + ringloss
