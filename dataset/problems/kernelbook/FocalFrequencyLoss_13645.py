"""Focal frequency loss: patchify -> fft2(ortho) -> spectrum weight matrix -> weighted mean squared frequency distance."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FocalFrequencyLoss_13645',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'loss_weight': 1.0,
    'alpha': 1.0,
    'patch_factor': 1,
    'ave_spectrum': False,
    'log_matrix': False,
    'batch_matrix': False,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return pred, target


def _tensor2freq(x):
    pf = CONFIG['patch_factor']
    n, c, h, w = x.shape
    ph, pw = h // pf, w // pf
    # (N, C, pf, ph, pf, pw) -> (N, pf, pf, C, ph, pw): row-major patch order like torch.stack
    y = x.reshape(n, c, pf, ph, pf, pw).transpose(0, 2, 4, 1, 3, 5).reshape(n, pf * pf, c, ph, pw)
    return jnp.fft.fft2(y, axes=(-2, -1), norm='ortho')


def workload(pred, target):
    pred_freq = _tensor2freq(pred)
    target_freq = _tensor2freq(target)
    if CONFIG['ave_spectrum']:
        pred_freq = jnp.mean(pred_freq, axis=0, keepdims=True)
        target_freq = jnp.mean(target_freq, axis=0, keepdims=True)
    diff = pred_freq - target_freq
    freq_distance = jnp.real(diff) ** 2 + jnp.imag(diff) ** 2

    matrix = jnp.sqrt(freq_distance) ** CONFIG['alpha']
    if CONFIG['log_matrix']:
        matrix = jnp.log(matrix + 1.0)
    if CONFIG['batch_matrix']:
        matrix = matrix / jnp.max(matrix)
    else:
        matrix = matrix / jnp.max(matrix, axis=(-2, -1), keepdims=True)
    matrix = jnp.where(jnp.isnan(matrix), 0.0, matrix)
    matrix = lax.stop_gradient(jnp.clip(matrix, 0.0, 1.0))

    return jnp.mean(matrix * freq_distance) * CONFIG['loss_weight']
