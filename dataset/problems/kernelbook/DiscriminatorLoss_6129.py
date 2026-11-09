"""[bf16 variant] DiscriminatorLoss: BCEWithLogits(fake, 0) and BCEWithLogits(real, 1), mean-reduced, then averaged."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiscriminatorLoss_6129',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_fake, k_real = jax.random.split(jax.random.key(0))
    fake_pred = jax.random.uniform(k_fake, shape, dtype)
    real_pred = jax.random.uniform(k_real, shape, dtype)
    return fake_pred, real_pred


def _bce_with_logits_mean(x, y):
    # Same numerically stable form PyTorch uses for BCEWithLogitsLoss.
    loss = jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x)))
    return jnp.mean(loss)


def workload(fake_pred, real_pred):
    fake_loss = _bce_with_logits_mean(fake_pred, jnp.zeros_like(fake_pred))
    real_loss = _bce_with_logits_mean(real_pred, jnp.ones_like(real_pred))
    return (fake_loss + real_loss) / 2
