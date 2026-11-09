"""[bf16 variant] MiniBatchStdDev: grouped unbiased var -> sqrt(+eps) -> mean -> broadcast -> concat on channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MiniBatchStdDev_4774',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'group_size': 4,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    g = CONFIG['group_size']
    # Matches torch x.view(group_size, -1): row-major flatten split into g rows.
    grouped = x.reshape(g, -1)
    # torch.var defaults to the unbiased estimator (correction=1).
    var = jnp.var(grouped, axis=0, ddof=1)
    std = jnp.mean(jnp.sqrt(var + CONFIG['eps']))
    b, _, h, w = x.shape
    std_map = jnp.broadcast_to(std.astype(x.dtype), (b, 1, h, w))
    return jnp.concatenate([x, std_map], axis=1)
