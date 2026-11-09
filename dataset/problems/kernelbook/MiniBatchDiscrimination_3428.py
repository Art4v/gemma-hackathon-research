"""[bf16 variant] Minibatch discrimination: x @ T -> pairwise L1 over kernel dims -> sum exp(-norm) - 1 -> concat with x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MiniBatchDiscrimination_3428',
    'batch': 256,
    'in_features': 8192,
    'out_features': 128,
    'kernel_dims': 16,
    'mean': False,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kt = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['in_features']), dtype)
    T = jax.random.normal(
        kt, (CONFIG['in_features'], CONFIG['out_features'], CONFIG['kernel_dims']), dtype) * 0.02
    return x, T


def workload(x, T):
    in_f, out_f, kd = T.shape
    m = (x @ T.reshape(in_f, out_f * kd)).reshape(-1, out_f, kd)
    # norm[j, i, o] = sum_k |M[i] - M[j]|
    norm = jnp.abs(m[None, :, :, :] - m[:, None, :, :]).sum(axis=3)
    o_b = jnp.exp(-norm).sum(axis=0) - 1
    if CONFIG['mean']:
        o_b = o_b / (x.shape[0] - 1)
    return jnp.concatenate([x, o_b], axis=1)
