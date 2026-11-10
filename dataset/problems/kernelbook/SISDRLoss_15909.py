"""SI-SDR loss: zero-mean -> projection scale alpha -> 10*log10 energy ratio -> mean -> negate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SISDRLoss_15909',
    'batch': 16,
    'nchs': 8,
    'time': 262144,
    'eps': 1e-8,
    'zero_mean': True,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['nchs'], CONFIG['time'])
    x = jax.random.uniform(k1, shape, dtype)
    t = jax.random.uniform(k2, shape, dtype)
    return x, t


def workload(input, target):
    eps = CONFIG['eps']
    if CONFIG['zero_mean']:
        input = input - jnp.mean(input, axis=-1, keepdims=True)
        target = target - jnp.mean(target, axis=-1, keepdims=True)
    alpha = jnp.sum(input * target, axis=-1) / (jnp.sum(target ** 2, axis=-1) + eps)
    target = target * alpha[..., None]
    res = input - target
    losses = 10 * jnp.log10(
        jnp.sum(target ** 2, axis=-1) / (jnp.sum(res ** 2, axis=-1) + eps) + eps
    )
    return -jnp.mean(losses)
