"""Pyramid pooling: sum of input and bilinear(align_corners=True)-upsampled adaptive avg pools at bins 1,2,3,6."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PyramidPoolingModule_16140',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'pyramids': (1, 2, 3, 6),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def _pool_matrix(out, size, dtype):
    # PyTorch adaptive pool bins: [floor(i*size/out), ceil((i+1)*size/out))
    idx = jnp.arange(out)
    start = (idx * size) // out
    end = ((idx + 1) * size + out - 1) // out
    pos = jnp.arange(size)
    mask = (pos[None, :] >= start[:, None]) & (pos[None, :] < end[:, None])
    return (mask / (end - start)[:, None]).astype(dtype)


def _interp_matrix(size, inp, dtype):
    # Bilinear with align_corners=True: src = i * (inp - 1) / (size - 1)
    scale = (inp - 1) / (size - 1) if size > 1 else 0.0
    src = jnp.arange(size, dtype=jnp.float32) * scale
    lo = jnp.clip(jnp.floor(src).astype(jnp.int32), 0, inp - 1)
    hi = jnp.minimum(lo + 1, inp - 1)
    frac = src - lo.astype(jnp.float32)
    m = (jax.nn.one_hot(lo, inp, dtype=jnp.float32) * (1.0 - frac)[:, None]
         + jax.nn.one_hot(hi, inp, dtype=jnp.float32) * frac[:, None])
    return m.astype(dtype)


def workload(x):
    height, width = x.shape[2], x.shape[3]
    hp = jax.lax.Precision.HIGHEST
    feat = x
    for b in CONFIG['pyramids']:
        ph = _pool_matrix(b, height, x.dtype)
        pw = _pool_matrix(b, width, x.dtype)
        pooled = jnp.einsum('ph,nchw,qw->ncpq', ph, x, pw, precision=hp)
        mh = _interp_matrix(height, b, x.dtype)
        mw = _interp_matrix(width, b, x.dtype)
        up = jnp.einsum('hp,ncpq,wq->nchw', mh, pooled, mw, precision=hp)
        feat = feat + up
    return feat
