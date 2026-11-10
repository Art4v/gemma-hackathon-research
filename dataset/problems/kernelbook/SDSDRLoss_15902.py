"""Scale-dependent SDR loss: zero-mean, projection scale alpha, log10 signal/residual ratio, mean reduction, negate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SDSDRLoss_15902',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'samples': 256,
    'zero_mean': True,
    'eps': 1e-08,
    'db_scale': 10.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['samples'])
    key = jax.random.key(0)
    k_input, k_target = jax.random.split(key)
    input_signal = jax.random.uniform(k_input, shape, dtype)
    target_signal = jax.random.uniform(k_target, shape, dtype)
    return input_signal, target_signal


def workload(input_signal, target_signal):
    eps = CONFIG['eps']
    if CONFIG['zero_mean']:
        input_signal = input_signal - jnp.mean(input_signal, axis=-1, keepdims=True)
        target_signal = target_signal - jnp.mean(target_signal, axis=-1, keepdims=True)
    alpha = (input_signal * target_signal).sum(-1) / ((target_signal ** 2).sum(-1) + eps)
    scaled_target = target_signal * alpha[..., None]
    res = input_signal - target_signal
    losses = CONFIG['db_scale'] * jnp.log10(
        (scaled_target ** 2).sum(-1) / ((res ** 2).sum(-1) + eps) + eps
    )
    return -losses.mean()
