"""Mix2Pooling: channel chunk -> AdaptiveAvgPool2d(first half) | AdaptiveMaxPool2d(second half) -> concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Mix2Pooling_4323',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'size': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def _bins(n, out):
    # PyTorch adaptive pooling: start = floor(i*n/out), end = ceil((i+1)*n/out).
    return [((i * n) // out, -((-(i + 1) * n) // out)) for i in range(out)]


def _adaptive_pool(x, size, reduce_fn):
    h, w = x.shape[2], x.shape[3]
    w_bins = _bins(w, size)
    rows = []
    for hs, he in _bins(h, size):
        cols = [reduce_fn(x[:, :, hs:he, ws:we], axis=(2, 3)) for ws, we in w_bins]
        rows.append(jnp.stack(cols, axis=-1))
    return jnp.stack(rows, axis=-2)


def workload(x):
    size = CONFIG['size']
    c0 = (x.shape[1] + 1) // 2  # torch.chunk gives the first chunk ceil(C/2) channels
    avg = _adaptive_pool(x[:, :c0], size, jnp.mean)
    mx = _adaptive_pool(x[:, c0:], size, jnp.max)
    return jnp.concatenate([avg, mx], axis=1)
