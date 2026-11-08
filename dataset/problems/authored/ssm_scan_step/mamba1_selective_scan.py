"""Mamba-1 selective scan: softplus(delta+bias) -> exp(delta*A) diagonal recurrence with input-dependent B,C (f32 state) -> C readout + D skip -> SiLU(z) gate."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mamba1_selective_scan',
    'batch': 2,
    'seqlen': 4096,
    'd_inner': 4096,
    'd_state': 16,
    'dt_init': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    L = CONFIG['seqlen']
    d = CONFIG['d_inner']
    n = CONFIG['d_state']
    key = jax.random.key(0)
    kx, kd, kb, kc, kz = jax.random.split(key, 5)
    x = jax.random.normal(kx, (b, L, d), dtype)
    delta_raw = jax.random.normal(kd, (b, L, d), dtype)
    B = jax.random.normal(kb, (b, L, n), dtype)
    C = jax.random.normal(kc, (b, L, n), dtype)
    z = jax.random.normal(kz, (b, L, d), dtype)
    # S4D-real initialisation, as in Mamba: A = -[1..N] per channel (fp32 params).
    A_log = jnp.log(jnp.broadcast_to(jnp.arange(1, n + 1, dtype=jnp.float32), (d, n)))
    D_skip = jnp.ones((d,), jnp.float32)
    dt = CONFIG['dt_init']
    dt_bias = jnp.full((d,), dt + jnp.log(-jnp.expm1(-dt)), jnp.float32)  # inverse softplus
    return (x, delta_raw, B, C, z, A_log, D_skip, dt_bias)


def workload(x, delta_raw, B, C, z, A_log, D_skip, dt_bias):
    out_dtype = x.dtype
    b, L, d = x.shape
    n = A_log.shape[-1]

    A = -jnp.exp(A_log.astype(jnp.float32))  # (d, n)
    xf = x.astype(jnp.float32)
    delta = jax.nn.softplus(delta_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32))
    Bf = B.astype(jnp.float32)
    Cf = C.astype(jnp.float32)

    # time-major for the scan
    xs = (
        jnp.swapaxes(xf, 0, 1),      # (L, b, d)
        jnp.swapaxes(delta, 0, 1),   # (L, b, d)
        jnp.swapaxes(Bf, 0, 1),      # (L, b, n)
        jnp.swapaxes(Cf, 0, 1),      # (L, b, n)
    )

    def step(h, inp):
        x_t, dt_t, B_t, C_t = inp
        dA = jnp.exp(dt_t[:, :, None] * A[None, :, :])                 # (b, d, n)
        dBx = (dt_t * x_t)[:, :, None] * B_t[:, None, :]               # (b, d, n)
        h = dA * h + dBx
        y_t = jnp.sum(h * C_t[:, None, :], axis=-1)                    # (b, d)
        return h, y_t

    h0 = jnp.zeros((b, d, n), jnp.float32)
    _, ys = lax.scan(step, h0, xs)
    y = jnp.swapaxes(ys, 0, 1)                                         # (b, L, d)
    y = y + xf * D_skip.astype(jnp.float32)[None, None, :]
    y = y * jax.nn.silu(z.astype(jnp.float32))
    return y.astype(out_dtype)
