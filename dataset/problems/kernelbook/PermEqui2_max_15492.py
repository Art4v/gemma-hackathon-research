"""[bf16 variant] PermEqui2_max: Gamma(x) - Lambda(max_over_set(x)), Linear layers on the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PermEqui2_max_15492',
    'batch': 16,
    'set_size': 128,
    'seq': 128,
    'in_dim': 256,
    'out_dim': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_gw, k_gb, k_lw = jax.random.split(jax.random.key(0), 4)
    c = CONFIG
    x = jax.random.uniform(k_x, (c['batch'], c['set_size'], c['seq'], c['in_dim']), dtype)
    gamma_w = jax.random.normal(k_gw, (c['in_dim'], c['out_dim']), dtype) * 0.02
    gamma_b = jax.random.normal(k_gb, (c['out_dim'],), dtype) * 0.02
    lambda_w = jax.random.normal(k_lw, (c['in_dim'], c['out_dim']), dtype) * 0.02
    return x, gamma_w, gamma_b, lambda_w


def workload(x, gamma_w, gamma_b, lambda_w):
    xm = jnp.max(x, axis=1, keepdims=True)
    xm = xm @ lambda_w
    y = x @ gamma_w + gamma_b
    return y - xm
