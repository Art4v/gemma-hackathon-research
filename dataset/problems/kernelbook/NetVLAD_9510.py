"""[bf16 variant] NetVLAD: L2-normalize -> 1x1 conv soft-assign -> softmax -> residual aggregation -> intra + global L2-normalize."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NetVLAD_9510',
    'batch': 8,
    'dim': 128,
    'height': 128,
    'width': 128,
    'num_clusters': 64,
    'alpha': 100.0,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kc = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']), dtype)
    centroids = jax.random.uniform(kc, (CONFIG['num_clusters'], CONFIG['dim']), dtype)
    alpha = CONFIG['alpha']
    # Mirrors NetVLAD._init_params: conv is tied to the centroids.
    conv_w = (2.0 * alpha * centroids).T  # (C, K)
    conv_b = -alpha * jnp.linalg.norm(centroids, axis=1)  # (K,)
    return x, conv_w, conv_b, centroids


def _l2_normalize(v, axis):
    norm = jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=True))
    return v / jnp.maximum(norm, CONFIG['eps'])


def workload(x, conv_w, conv_b, centroids):
    n, c, h, w = x.shape
    hi = lax.Precision.HIGHEST
    xf = _l2_normalize(x, 1).reshape(n, c, h * w)
    logits = jnp.einsum('ck,ncp->nkp', conv_w, xf, precision=hi) + conv_b[None, :, None]
    a = jax.nn.softmax(logits, axis=1)
    # sum_p a*(x - cent) == a@x^T - cent*sum_p a; avoids the (N,K,C,HW) residual tensor.
    vlad = jnp.einsum('nkp,ncp->nkc', a, xf, precision=hi) - centroids[None] * jnp.sum(a, axis=-1)[..., None]
    vlad = _l2_normalize(vlad, 2).reshape(n, -1)
    return _l2_normalize(vlad, 1)
