"""[bf16 variant] ContrastiveLoss: sum_c (x0-x1)^2 -> sqrt -> clamp(margin - d, 0) -> y*d^2 + (1-y)*hinge^2 -> sum / 2 / B."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_10809',
    'batch': 32,
    'channels': 32,  # must equal batch: PyTorch broadcasts dist_sq (B,H,W) against y (B,C,H,W)
    'height': 128,
    'width': 128,
    'margin': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x0 = jax.random.uniform(k0, shape, dtype)
    x1 = jax.random.uniform(k1, shape, dtype)
    y = jax.random.uniform(k2, shape, dtype)
    return x0, x1, y


def workload(x0, x1, y):
    diff = x0 - x1
    dist_sq = jnp.sum(diff * diff, axis=1)
    hinge = jnp.maximum(CONFIG['margin'] - jnp.sqrt(dist_sq), 0.0)
    # Trailing-dim broadcast (B,H,W) vs (B,C,H,W), same as PyTorch.
    loss = y * dist_sq + (1.0 - y) * hinge * hinge
    return jnp.sum(loss) / 2.0 / x0.shape[0]
