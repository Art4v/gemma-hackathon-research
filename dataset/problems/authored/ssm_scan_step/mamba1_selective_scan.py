"""Mamba-1 selective scan: softplus(delta+bias) -> discretize diag A, input-dependent B/C -> f32 recurrent scan -> C readout + D skip -> SiLU(z) gate."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mamba1_selective_scan',
    'batch': 4,
    'seqlen': 2048,
    'd_inner': 2048,
    'd_state': 16,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    l = CONFIG['seqlen']
    d = CONFIG['d_inner']
    n = CONFIG['d_state']
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(keys[0], (b, l, d), dtype)
    delta = jax.random.normal(keys[1], (b, l, d), dtype)
    z = jax.random.normal(keys[2], (b, l, d), dtype)
    B = jax.random.normal(keys[3], (b, l, n), dtype)
    C = jax.random.normal(keys[4], (b, l, n), dtype)
    # S4D-real initialization: A = -[1..N] per channel, stored as log.
    A_log = jnp.log(jnp.broadcast_to(jnp.arange(1, n + 1, dtype=jnp.float32), (d, n)))
    D = jnp.ones((d,), jnp.float32)
    delta_bias = jax.random.normal(keys[5], (d,), jnp.float32) * CONFIG['weight_scale']
    return x, delta, z, B, C, A_log, D, delta_bias


def workload(x, delta, z, B, C, A_log, D, delta_bias):
    out_dtype = x.dtype
    bsz, _, d = x.shape
    n = A_log.shape[-1]

    xf = x.astype(jnp.float32)
    dt = jax.nn.softplus(delta.astype(jnp.float32) + delta_bias.astype(jnp.float32))
    A = -jnp.exp(A_log.astype(jnp.float32))  # (D, N)
    Bf = B.astype(jnp.float32)
    Cf = C.astype(jnp.float32)

    # Time-major for the scan.
    xs = (
        jnp.swapaxes(xf, 0, 1),   # (L, B, D)
        jnp.swapaxes(dt, 0, 1),   # (L, B, D)
        jnp.swapaxes(Bf, 0, 1),   # (L, B, N)
        jnp.swapaxes(Cf, 0, 1),   # (L, B, N)
    )

    def step(h, inp):
        x_t, dt_t, b_t, c_t = inp
        dA = jnp.exp(dt_t[:, :, None] * A[None, :, :])               # (B, D, N)
        dBx = (dt_t * x_t)[:, :, None] * b_t[:, None, :]             # (B, D, N)
        h = dA * h + dBx
        y_t = jnp.einsum('bdn,bn->bd', h, c_t)                       # (B, D)
        return h, y_t

    h0 = jnp.zeros((bsz, d, n), jnp.float32)
    _, ys = lax.scan(step, h0, xs)
    y = jnp.swapaxes(ys, 0, 1)                                       # (B, L, D)
    y = y + xf * D.astype(jnp.float32)[None, None, :]
    y = y * jax.nn.silu(z.astype(jnp.float32))
    return y.astype(out_dtype)
