"""GeneratorLoss: BCEWithLogits(fake_pred, ones) + alpha * L1(fake, real), mean reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeneratorLoss_6142',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'alpha': 100.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_fake, k_real, k_pred = jax.random.split(key, 3)
    fake = jax.random.uniform(k_fake, shape, dtype)
    real = jax.random.uniform(k_real, shape, dtype)
    fake_pred = jax.random.uniform(k_pred, shape, dtype)
    return fake, real, fake_pred


def workload(fake, real, fake_pred):
    fake_target = jnp.ones_like(fake_pred)
    bce = jnp.mean(
        (1.0 - fake_target) * fake_pred
        + jnp.logaddexp(jnp.zeros_like(fake_pred), -fake_pred)
    )
    l1 = jnp.mean(jnp.abs(fake - real))
    return bce + CONFIG['alpha'] * l1
