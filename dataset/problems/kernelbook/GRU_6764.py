"""[bf16 variant] GRU cell step: Linear(x)+Linear(h) -> sigmoid gates -> tanh candidate -> gated update -> mask blend (broadcast as in PyTorch)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GRU_6764',
    'batch': 256,
    'input_size': 2048,
    'hidden_size': 2048,
}


def create_inputs(dtype=jnp.bfloat16):
    b, i, h = CONFIG['batch'], CONFIG['input_size'], CONFIG['hidden_size']
    ks = jax.random.split(jax.random.key(0), 11)
    x_t = jax.random.uniform(ks[0], (b, i), dtype)
    x_m = jax.random.uniform(ks[1], (b, h), dtype)
    h_tm1 = jax.random.uniform(ks[2], (b, h), dtype)
    w_xh = jax.random.normal(ks[3], (i, h), dtype) * 0.02
    b_xh = jax.random.normal(ks[4], (h,), dtype) * 0.02
    w_hh = jax.random.normal(ks[5], (h, h), dtype) * 0.02
    b_hh = jax.random.normal(ks[6], (h,), dtype) * 0.02
    w_xrz = jax.random.normal(ks[7], (i, 2 * h), dtype) * 0.02
    b_xrz = jax.random.normal(ks[8], (2 * h,), dtype) * 0.02
    w_hrz = jax.random.normal(ks[9], (h, 2 * h), dtype) * 0.02
    b_hrz = jax.random.normal(ks[10], (2 * h,), dtype) * 0.02
    return (x_t, x_m, h_tm1, w_xh, b_xh, w_hh, b_hh, w_xrz, b_xrz, w_hrz, b_hrz)


def workload(x_t, x_m, h_tm1, w_xh, b_xh, w_hh, b_hh, w_xrz, b_xrz, w_hrz, b_hrz):
    hidden = h_tm1.shape[-1]
    rz_t = jax.nn.sigmoid((x_t @ w_xrz + b_xrz) + (h_tm1 @ w_hrz + b_hrz))
    r_t, z_t = rz_t[:, :hidden], rz_t[:, hidden:]
    h_above = jnp.tanh((x_t @ w_xh + b_xh) + ((r_t * h_tm1) @ w_hh + b_hh))
    h_t = (1.0 - z_t) * h_tm1 + z_t * h_above
    # x_m is 2-D, so x_m[:, None] is (B,1,H) and broadcasts to (B,B,H), exactly as in the PyTorch module.
    return x_m[:, None] * h_t + (1.0 - x_m[:, None]) * h_tm1
