"""[bf16 variant] TotalVariation: sum |x[:, :, 1:] - x[:, :, :-1] + eps| + |x[:, 1:] - x[:, :-1] + eps| over axes 0-2, divided by numel."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TotalVariation_5718',
    'dim0': 8,
    'dim1': 128,
    'dim2': 256,
    'dim3': 256,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['dim0'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    adv_patch = jax.random.uniform(key, shape, dtype=dtype)
    return (adv_patch,)


def workload(adv_patch):
    eps = CONFIG['eps']
    # Torch reduces dim 0 three times, leaving only the last axis.
    tv1 = jnp.sum(jnp.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + eps), axis=(0, 1, 2))
    tv2 = jnp.sum(jnp.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + eps), axis=(0, 1, 2))
    return (tv1 + tv2) / adv_patch.size
