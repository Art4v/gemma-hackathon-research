"""[bf16 variant] Masked average pooling: masked_fill(mask==0) -> sum over axis 1 -> divide by clamp(count(mask==1), 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AveragePooling_7272',
    'num_items': 16,
    'max_subitem_size': 64,
    'feat_dim': 128,
    'input_size': 512,
    'min_count': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, km = jax.random.split(jax.random.key(0))
    a, b = CONFIG['num_items'], CONFIG['max_subitem_size']
    d, c = CONFIG['feat_dim'], CONFIG['input_size']
    x = jax.random.uniform(kx, (a, b, d, c), dtype=dtype)
    x_mask = jax.random.randint(km, (a, b, c), 0, 2).astype(dtype)
    return x, x_mask


def workload(x, x_mask):
    # mask (A,B,C) broadcasts as (A,B,1,C) over x (A,B,D,C), as in the torch unsqueeze(2)
    keep = (x_mask != 0)[:, :, None, :]
    x_sum = jnp.sum(jnp.where(keep, x, jnp.zeros((), x.dtype)), axis=1)
    x_num = jnp.sum((x_mask == 1).astype(jnp.float32), axis=1)[:, None, :]
    x_num = jnp.maximum(x_num, CONFIG['min_count'])
    return x_sum / x_num
