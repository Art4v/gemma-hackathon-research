"""MergeLayer: two Linear projections -> concat -> Linear(2H,1) -> sigmoid gate -> convex blend."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MergeLayer_2910',
    'batch': 16384,
    'h_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    b, h = CONFIG['batch'], CONFIG['h_size']
    k_in, k_out, k_wi, k_wo, k_wl, k_bl = jax.random.split(jax.random.key(0), 6)
    h_inbound = jax.random.uniform(k_in, (b, h), dtype)
    h_outbound = jax.random.uniform(k_out, (b, h), dtype)
    w_in = jax.random.normal(k_wi, (h, h), dtype) * 0.02
    b_in = jnp.zeros((h,), dtype)
    w_out = jax.random.normal(k_wo, (h, h), dtype) * 0.02
    b_out = jnp.zeros((h,), dtype)
    w_lam = jax.random.normal(k_wl, (2 * h, 1), dtype) * 0.02
    b_lam = jax.random.normal(k_bl, (1,), dtype) * 0.02
    return h_inbound, h_outbound, w_in, b_in, w_out, b_out, w_lam, b_lam


def workload(h_inbound, h_outbound, w_in, b_in, w_out, b_out, w_lam, b_lam):
    hi = h_inbound @ w_in + b_in
    ho = h_outbound @ w_out + b_out
    lam = jax.nn.sigmoid(jnp.concatenate([hi, ho], axis=1) @ w_lam + b_lam)
    return lam * hi + (1 - lam) * ho
