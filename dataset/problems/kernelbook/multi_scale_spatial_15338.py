"""Limb row gather -> AdaptiveMaxPool2d((1,20)) x5 -> concat -> AdaptiveAvgPool2d((1,20))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'multi_scale_spatial_15338',
    'batch': 128,
    'channels': 512,
    'height': 8,
    'width': 128,
    # order matches the torch module: left_arm, right_arm, left_leg, right_leg, head_spine
    'limb_blocks': (4, 4, 4, 4, 4),
    'pool_w': 20,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _adaptive_last(t, out, reduce_fn):
    # PyTorch adaptive pooling windows: [floor(i*L/out), ceil((i+1)*L/out))
    length = t.shape[-1]
    bins = [
        reduce_fn(t[..., (i * length) // out:-(-((i + 1) * length) // out)], axis=-1)
        for i in range(out)
    ]
    return jnp.stack(bins, axis=-1)


def _limb_maxpool(x, idx, out):
    # x[:, :, idx] -> (N, C, W); torch treats it as unbatched (C', H', W),
    # so adaptive H'->1 is a max over the full C axis.
    t = jnp.max(x[:, :, idx], axis=1, keepdims=True)
    return _adaptive_last(t, out, jnp.max)


def workload(x):
    left_arm, right_arm, left_leg, right_leg, head_spine = CONFIG['limb_blocks']
    out = CONFIG['pool_w']
    parts = [_limb_maxpool(x, i, out) for i in (left_leg, right_leg, left_arm, right_arm, head_spine)]
    multi = jnp.concatenate(parts, axis=-2)  # (N, 5, out)
    pooled = jnp.mean(multi, axis=-2, keepdims=True)  # adaptive H 5->1
    return _adaptive_last(pooled, out, jnp.mean)
