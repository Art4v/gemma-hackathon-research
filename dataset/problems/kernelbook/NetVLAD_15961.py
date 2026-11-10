"""[bf16 variant] NetVLAD: L2 normalize -> 1x1 conv + bias -> softmax -> soft-assigned residual sum -> per-cluster L2 normalize -> flatten -> L2 normalize."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NetVLAD_15961',
    'batch': 8,
    'dim': 128,
    'height': 256,
    'width': 256,
    'num_clusters': 64,
    'alpha': 100.0,
    'normalize_input': True,
    'eps': 1e-12,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_c = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    centroids = jax.random.normal(
        k_c, (CONFIG['num_clusters'], CONFIG['dim']), dtype
    ) * CONFIG['weight_scale']
    conv_weight = (2.0 * CONFIG['alpha'] * centroids)[:, :, None, None]
    conv_bias = -CONFIG['alpha'] * jnp.sqrt(jnp.sum(centroids * centroids, axis=1))
    return x, conv_weight, conv_bias, centroids


def _normalize(v, axis):
    norm = jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=True))
    return v / jnp.maximum(norm, CONFIG['eps'])


def workload(x, conv_weight, conv_bias, centroids):
    N, C = x.shape[:2]
    K = conv_weight.shape[0]
    if CONFIG['normalize_input']:
        x = _normalize(x, 1)
    logits = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        conv_weight,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    logits = jnp.transpose(logits + conv_bias, (0, 3, 1, 2))
    soft_assign = jax.nn.softmax(logits.reshape(N, K, -1), axis=1)
    x_flatten = x.reshape(N, C, -1)
    # Equals sum_p (x_p - c_k) * a_kp without materializing the (N, K, C, P) residual.
    vlad = soft_assign @ jnp.swapaxes(x_flatten, 1, 2)
    vlad = vlad - centroids[None, :, :] * jnp.sum(soft_assign, axis=-1, keepdims=True)
    vlad = _normalize(vlad, 2)
    vlad = vlad.reshape(N, -1)
    return _normalize(vlad, 1)
