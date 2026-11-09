"""[bf16 variant] MixLoss (default weights): bce_w * BCEWithLogitsLoss(mean); dice/focal/lovasz weights are 0."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MixLoss_15978',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'bce_w': 1.0,
    # ponytail: zero by default in the torch module, so those branches never run.
    'dice_w': 0.0,
    'focal_w': 0.0,
    'lovasz_w': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    x = output.astype(jnp.float32)
    t = target.astype(jnp.float32)
    # Stable BCE-with-logits, same formulation as PyTorch.
    loss = jnp.maximum(x, 0.0) - x * t + jnp.log1p(jnp.exp(-jnp.abs(x)))
    return CONFIG['bce_w'] * jnp.mean(loss)
