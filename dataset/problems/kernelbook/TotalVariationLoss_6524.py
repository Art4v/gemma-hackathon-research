"""TotalVariationLoss: TV loss (mean((dh^2 + dw^2)^1.25)) as a side value; forward returns x unchanged."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TotalVariationLoss_6524',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
    'tv_power': 1.25,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def _tv_loss(x):
    h, w = x.shape[2], x.shape[3]
    base = x[:, :, :h - 1, :w - 1]
    a = jnp.square(base - x[:, :, 1:, :w - 1])
    b = jnp.square(base - x[:, :, :h - 1, 1:])
    return jnp.mean(jnp.power(a + b, CONFIG['tv_power']))


def workload(x):
    # ponytail: torch stores the loss on self.loss and returns only x; JAX has no
    # module state, so the loss is computed but not returned (XLA eliminates it).
    _ = _tv_loss(x)
    return x
