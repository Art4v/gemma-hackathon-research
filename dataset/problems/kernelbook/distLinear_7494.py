"""[bf16 variant] distLinear: L2-normalize over dim 1 -> weight-normed Linear (no bias) on last dim -> scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'distLinear_7494',
    'batch': 8,
    'channels': 128,
    'height': 64,
    'indim': 512,
    'outdim': 1024,
    'eps': 1e-5,
    'small_outdim_threshold': 200,
    'small_scale': 2.0,
    'large_scale': 10.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_v = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['indim']), dtype)
    v = jax.random.normal(k_v, (CONFIG['indim'], CONFIG['outdim']), dtype) * 0.02
    g = jnp.ones((CONFIG['outdim'],), dtype)
    return x, v, g


def workload(x, v, g):
    outdim = v.shape[1]
    scale = (CONFIG['small_scale'] if outdim <= CONFIG['small_outdim_threshold']
             else CONFIG['large_scale'])
    x_norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    x_normalized = x / (x_norm + CONFIG['eps'])
    # WeightNorm dim=0 in torch = per output row; v here is (in, out) so norm over axis 0.
    w = v * (g / jnp.sqrt(jnp.sum(v * v, axis=0)))[None, :]
    cos_dist = jnp.matmul(x_normalized, w, precision=lax.Precision.HIGHEST)
    return scale * cos_dist
