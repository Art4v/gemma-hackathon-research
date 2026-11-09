"""AdaIN2d: Linear(h) -> chunk(b, s) -> InstanceNorm2d(x) -> x * (s + 1) + b (torch trailing-axis broadcast)."""
import jax
import jax.numpy as jnp
from jax import lax

# ponytail: torch discards h.view(bs, fs, 1, 1), so (B, C)-shaped s/b broadcast
# against x's trailing (H, W) axes; height must equal batch and width must equal channels.
CONFIG = {
    'name': 'AdaIN2d_2953',
    'batch': 64,
    'channels': 128,
    'height': 64,
    'width': 128,
    'in_features': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    b, c, h, w, f = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'],
                     CONFIG['width'], CONFIG['in_features'])
    k_x, k_h, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    hin = jax.random.uniform(k_h, (b, f), dtype)
    weight = jax.random.normal(k_w, (f, 2 * c), dtype) * 0.02
    bias = jax.random.normal(k_b, (2 * c,), dtype) * 0.02
    return x, hin, weight, bias


def workload(x, h, weight, bias):
    y = jnp.dot(h, weight, precision=lax.Precision.HIGHEST) + bias
    c = y.shape[1] // 2
    shift, scale = y[:, :c], y[:, c:]
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(2, 3), keepdims=True)
    xn = (x - mean) * lax.rsqrt(var + CONFIG['eps'])
    # (B, C) broadcasts against trailing (H, W), exactly as in the torch module.
    return xn * (scale + 1) + shift
