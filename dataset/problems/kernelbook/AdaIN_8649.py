"""[bf16 variant] AdaIN: Linear(style) -> chunk(gamma, beta) -> (1 + gamma) * InstanceNorm2d(x) + beta."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdaIN_8649',
    'batch': 16,
    'num_features': 256,
    'height': 128,
    'width': 128,
    'style_dim': 512,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ks, kw, kb = jax.random.split(jax.random.key(0), 4)
    n, c = CONFIG['batch'], CONFIG['num_features']
    h, w, sd = CONFIG['height'], CONFIG['width'], CONFIG['style_dim']
    x = jax.random.uniform(kx, (n, c, h, w), dtype)
    s = jax.random.uniform(ks, (n, sd), dtype)
    fc_w = jax.random.normal(kw, (sd, 2 * c), dtype) * 0.02
    fc_b = jax.random.normal(kb, (2 * c,), dtype) * 0.02
    return x, s, fc_w, fc_b


def workload(x, s, fc_w, fc_b):
    c = x.shape[1]
    h = s @ fc_w + fc_b
    gamma = h[:, :c, None, None]
    beta = h[:, c:, None, None]
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(2, 3), keepdims=True)
    x_hat = (x - mean) * jax.lax.rsqrt(var + CONFIG['eps'])
    return (1.0 + gamma) * x_hat + beta
