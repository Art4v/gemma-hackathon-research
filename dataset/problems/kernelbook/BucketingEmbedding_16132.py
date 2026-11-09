"""BucketingEmbedding: linear bucketize (scale, clamp, truncate) followed by an embedding gather."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BucketingEmbedding_16132',
    'x_shape': (16, 128, 128),
    'min_val': 4.0,
    'max_val': 4.0,
    'count': 131072,
    'dim': 128,
    'use_log_scale': False,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, CONFIG['x_shape'], dtype)
    table = jax.random.normal(kw, (CONFIG['count'], CONFIG['dim']), dtype) * 0.02
    return x, table


def workload(x, table):
    count = table.shape[0]
    min_val = CONFIG['min_val']
    max_val = CONFIG['max_val']
    if CONFIG['use_log_scale']:
        x = jnp.log2(x)
        min_val = float(jnp.log2(jnp.float32(min_val)))
        max_val = float(jnp.log2(jnp.float32(max_val)))
    # min_val == max_val (as in the original init) gives -inf here, which clamps to bucket 0, matching torch.
    v = count * (x - min_val) / (max_val - min_val)
    idx = jnp.clip(v, 0, count - 1).astype(jnp.int32)
    return jnp.take(table, idx, axis=0)
