"""[bf16 variant] MMD loss: concat -> pairwise squared L2 -> multi-bandwidth Gaussian kernel sum -> block means (XX + YY - XY - YX)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MMD_4913',
    'source_num': 8192,
    'target_num': 8192,
    'dim': 2048,
    'kernel_mul': 2.0,
    'kernel_num': 5,
}


def create_inputs(dtype=jnp.bfloat16):
    k_src, k_tgt = jax.random.split(jax.random.key(0))
    source = jax.random.uniform(k_src, (CONFIG['source_num'], CONFIG['dim']), dtype)
    target = jax.random.uniform(k_tgt, (CONFIG['target_num'], CONFIG['dim']), dtype)
    return source, target


def workload(source, target):
    kernel_mul = CONFIG['kernel_mul']
    kernel_num = CONFIG['kernel_num']
    source_num = source.shape[0]
    target_num = target.shape[0]
    n_samples = source_num + target_num

    total = jnp.concatenate([source, target], axis=0)
    # Gram form replaces the (n, n, D) broadcast in the PyTorch module, which would need about 2 TB at these sizes.
    sq = jnp.sum(total * total, axis=1)
    gram = jnp.matmul(total, total.T, precision=lax.Precision.HIGHEST)
    l2 = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)

    bandwidth = lax.stop_gradient(jnp.sum(l2)) / (n_samples ** 2 - n_samples)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = sum(jnp.exp(-l2 / (bandwidth * kernel_mul ** i)) for i in range(kernel_num))

    xx = jnp.mean(kernels[:source_num, :source_num])
    yy = jnp.mean(kernels[target_num:, target_num:])
    xy = jnp.mean(kernels[:target_num, source_num:])
    yx = jnp.mean(kernels[source_num:, :target_num])
    return xx + yy - xy - yx
