"""BoF pooling: channel-sum + codeword sq-norm - 2 * 1x1 conv -> relu -> softmax(-d/sigma^2) over codewords -> spatial mean * N_k."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BoF_Pooling_13127',
    'batch': 16,
    'features': 512,
    'height': 64,
    'width': 64,
    'n_codewords': 256,
    'sigma_init': 0.1,
    'spatial_level': 0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_v = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['features'], CONFIG['height'], CONFIG['width']), dtype)
    v = jax.random.normal(k_v, (CONFIG['features'], CONFIG['n_codewords']), dtype) * 0.02
    sigmas = jnp.full((CONFIG['n_codewords'],), CONFIG['sigma_init'], dtype)
    return x, v, sigmas


def workload(x, v, sigmas):
    # x: (B, C, H, W) NCHW; v: (C, N_k) codebook; sigmas: (N_k,)
    n_k = v.shape[1]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Original sums the input (not its square) over channels; kept for parity.
    x_square = jnp.sum(x_nhwc, axis=-1, keepdims=True)
    y_square = jnp.sum(v ** 2, axis=0)
    xv = jnp.einsum('bhwc,ck->bhwk', x_nhwc, v, precision=lax.Precision.HIGHEST)
    dists = jax.nn.relu(x_square + y_square - 2.0 * xv)
    q = jax.nn.softmax(-dists / sigmas ** 2, axis=-1)
    # ponytail: spatial_level=0 only (the configured case); level 1 = 4 quadrant means concatenated.
    histogram = jnp.mean(q, axis=(1, 2))
    return histogram * n_k
