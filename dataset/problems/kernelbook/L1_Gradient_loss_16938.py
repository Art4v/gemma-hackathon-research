"""L1 gradient loss: H/W forward differences of X and Y, Charbonnier sqrt(d^2+eps) mean per direction, averaged."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L1_Gradient_loss_16938',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'eps': 1e-6,
    'scale': 0.5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, ky = jax.random.split(jax.random.key(0))
    X = jax.random.uniform(kx, shape, dtype)
    Y = jax.random.uniform(ky, shape, dtype)
    return X, Y


def _charbonnier(a, b):
    diff = a - b
    return jnp.mean(jnp.sqrt(diff * diff + CONFIG['eps']))


def workload(X, Y):
    xgin = X[:, :, 1:, :] - X[:, :, :-1, :]
    ygin = X[:, :, :, 1:] - X[:, :, :, :-1]
    xgtarget = Y[:, :, 1:, :] - Y[:, :, :-1, :]
    ygtarget = Y[:, :, :, 1:] - Y[:, :, :, :-1]
    xl = _charbonnier(xgin, xgtarget)
    yl = _charbonnier(ygin, ygtarget)
    return (xl + yl) * CONFIG['scale']
