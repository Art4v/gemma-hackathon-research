"""DiscriminatorLoss: mean(BCEWithLogits(fake, 0)) and mean(BCEWithLogits(real, 1)), averaged."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiscriminatorLoss_18296',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'fake_target': 0.0,
    'real_target': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_fake, k_real = jax.random.split(jax.random.key(0))
    fake_pred = jax.random.uniform(k_fake, shape, dtype)
    real_pred = jax.random.uniform(k_real, shape, dtype)
    return fake_pred, real_pred


def _bce_with_logits_mean(x, y):
    # Same stable form PyTorch uses: max(x,0) - x*y + log1p(exp(-|x|)).
    loss = jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x)))
    return jnp.mean(loss)


def workload(fake_pred, real_pred):
    fake_loss = _bce_with_logits_mean(fake_pred, jnp.full_like(fake_pred, CONFIG['fake_target']))
    real_loss = _bce_with_logits_mean(real_pred, jnp.full_like(real_pred, CONFIG['real_target']))
    return (fake_loss + real_loss) / 2
