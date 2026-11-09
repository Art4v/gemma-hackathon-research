"""PatchMerging: 2x2 strided gather + channel concat -> LayerNorm over last axis -> 1x1 conv (4C -> 2C, no bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PatchMerging_16516',
    'batch': 8,
    'dim': 64,
    'height': 256,
    'width': 512,  # width // 2 must equal 4 * dim: LayerNorm(4*dim) acts on the last (W/2) axis
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    b, c, h, w = CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    reduction_w = jax.random.normal(k_w, (4 * c, 2 * c), dtype) * 0.02
    norm_gamma = jnp.ones((4 * c,), dtype)
    norm_beta = jnp.zeros((4 * c,), dtype)
    return x, reduction_w, norm_gamma, norm_beta


def workload(x, reduction_w, norm_gamma, norm_beta):
    x0 = x[:, :, 0::2, 0::2]
    x1 = x[:, :, 1::2, 0::2]
    x2 = x[:, :, 0::2, 1::2]
    x3 = x[:, :, 1::2, 1::2]
    y = jnp.concatenate([x0, x1, x2, x3], axis=1)
    # PyTorch LayerNorm(4*dim) on an NCHW tensor normalizes the last axis (W/2), not channels.
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    y = (y - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * norm_gamma + norm_beta
    # 1x1 conv without bias == per-pixel channel matmul.
    return jnp.einsum('bchw,co->bohw', y, reduction_w)
