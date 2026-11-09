"""[bf16 variant] BCEBlurWithLogitsLoss: BCEWithLogits(reduction='none') * (1 - exp((sigmoid(pred) - true - 1) / (alpha + 1e-4))), then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEBlurWithLogitsLoss_256',
    'batch': 8,
    'channels': 32,
    'height': 512,
    'width': 512,
    'alpha': 0.05,
    'alpha_eps': 0.0001,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_true = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k_pred, shape, dtype)
    true = jax.random.uniform(k_true, shape, dtype)
    return pred, true


def workload(pred, true):
    # Stable BCE with logits, matching torch's formulation.
    loss = jnp.maximum(pred, 0) - pred * true + jnp.log1p(jnp.exp(-jnp.abs(pred)))
    dx = jax.nn.sigmoid(pred) - true
    alpha_factor = 1 - jnp.exp((dx - 1) / (CONFIG['alpha'] + CONFIG['alpha_eps']))
    return jnp.mean(loss * alpha_factor)
