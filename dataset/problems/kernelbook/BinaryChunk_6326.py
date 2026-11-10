"""[bf16 variant] BinaryChunk: torch.chunk on last dim, chunk on dim -2, per-sample flatten + max pool, stack."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryChunk_6326',
    'n_cls': 4,
    'n_chunk': 2,  # ceil(log2(n_cls))
    'n_split': 2,
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _chunk_bounds(size, n):
    step = -(-size // n)
    return [(start, min(start + step, size)) for start in range(0, size, step)]


def workload(x):
    n_samp = x.shape[0]
    pooled = []
    for w0, w1 in _chunk_bounds(x.shape[-1], CONFIG['n_chunk']):
        ck = x[..., w0:w1]
        for h0, h1 in _chunk_bounds(ck.shape[-2], CONFIG['n_split']):
            xx = ck[..., h0:h1, :]
            pooled.append(jnp.max(xx.reshape(n_samp, -1), axis=1))
    return jnp.stack(pooled, axis=1)
