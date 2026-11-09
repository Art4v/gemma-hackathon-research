"""[bf16 variant] Linear(no bias) -> LayerNorm(no affine) -> ReLU -> Linear -> chunk -> (mu, softplus(sigma+0.5)+1e-8)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussianParamNet_4116',
    'batch': 16,
    'k': 1024,
    'input_dim': 1024,
    'output_dim': 1024,
    'ln_eps': 1e-5,
    'sigma_shift': 0.5,
    'sigma_floor': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 4)
    d, o = CONFIG['input_dim'], CONFIG['output_dim']
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['k'], d), dtype)
    w1 = jax.random.normal(k_w1, (d, d), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (d, o), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (o,), dtype) * 0.02
    return x, w1, w2, b2


def workload(x, w1, w2, b2):
    h = x @ w1
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.var(h, axis=-1, keepdims=True)
    h = (h - mean) * jax.lax.rsqrt(var + CONFIG['ln_eps'])
    h = jax.nn.relu(h)
    y = h @ w2 + b2
    mu, sigma = jnp.split(y, 2, axis=-1)
    sigma = jax.nn.softplus(sigma + CONFIG['sigma_shift']) + CONFIG['sigma_floor']
    return mu, sigma
