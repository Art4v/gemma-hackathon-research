"""[bf16 variant] Multiscale L1 loss: weighted sum of mean |input - target| over 5 scales of 2x2 average pooling."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiscaleL1Loss_18116',
    'batch': 16,
    'channels': 4,
    'height': 1024,
    'width': 1024,
    'pool': 2,
    'weights': (1.0, 0.5, 0.25, 0.125, 0.125),
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, shape, dtype)
    y = jax.random.uniform(k2, shape, dtype)
    return x, y


def _avg_pool(x):
    # AvgPool2d(k, stride=k, padding=0): floor-crop, then mean over non-overlapping k x k windows.
    p = CONFIG['pool']
    b, c, h, w = x.shape
    ho, wo = h // p, w // p
    x = x[:, :, :ho * p, :wo * p]
    return x.reshape(b, c, ho, p, wo, p).mean(axis=(3, 5))


def workload(x, y):
    weights = CONFIG['weights']
    loss = jnp.zeros((), dtype=x.dtype)
    for i, wt in enumerate(weights):
        loss = loss + wt * jnp.mean(jnp.abs(x - y))
        if i != len(weights) - 1:
            x = _avg_pool(x)
            y = _avg_pool(y)
    return loss
