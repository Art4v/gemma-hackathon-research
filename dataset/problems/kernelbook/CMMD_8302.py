"""CMMD loss: flatten -> L2 normalize -> split by num_pos -> squared-dot polynomial-kernel MMD per chunk -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CMMD_8302',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'num_pos': 4,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_v, key_t = jax.random.split(jax.random.key(0))
    feat_v = jax.random.uniform(key_v, shape, dtype)
    feat_t = jax.random.uniform(key_t, shape, dtype)
    return feat_v, feat_t


def _normalize(x):
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
    return x / jnp.maximum(norm, CONFIG['eps'])


def _split(x):
    size = CONFIG['num_pos']
    return [x[i:i + size] for i in range(0, x.shape[0], size)]


def _poly_kernel_mean(a, b):
    return jnp.mean(jnp.square(a @ b.T))


def _mmd_loss(f_v, f_t):
    return (
        _poly_kernel_mean(f_v, f_v)
        + _poly_kernel_mean(f_t, f_t)
        - 2 * _poly_kernel_mean(f_v, f_t)
    )


def workload(feat_v, feat_t):
    v = _normalize(feat_v.reshape(feat_v.shape[0], -1))
    t = _normalize(feat_t.reshape(feat_t.shape[0], -1))
    losses = [_mmd_loss(f_v, f_t) for f_v, f_t in zip(_split(v), _split(t))]
    return sum(losses) / len(losses)
