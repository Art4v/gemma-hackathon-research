"""[bf16 variant] PartitionedLinear: chunk(x, 2, -1) -> (linear_c(x_c), linear_p(x_p))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PartitionedLinear_4357',
    'batch': 8,
    'seq_len': 2048,
    'in_features': 4096,
    'out_features': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    b, s = CONFIG['batch'], CONFIG['seq_len']
    in_h = CONFIG['in_features'] // 2
    out_h = CONFIG['out_features'] // 2
    k_x, k_wc, k_bc, k_wp, k_bp = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (b, s, CONFIG['in_features']), dtype)
    w_c = jax.random.normal(k_wc, (in_h, out_h), dtype) * 0.02
    b_c = jax.random.normal(k_bc, (out_h,), dtype) * 0.02
    w_p = jax.random.normal(k_wp, (in_h, out_h), dtype) * 0.02
    b_p = jax.random.normal(k_bp, (out_h,), dtype) * 0.02
    return x, w_c, b_c, w_p, b_p


def workload(x, w_c, b_c, w_p, b_p):
    half = x.shape[-1] // 2
    x_c = x[..., :half]
    x_p = x[..., half:]
    out_c = x_c @ w_c + b_c
    out_p = x_p @ w_p + b_p
    return out_c, out_p
