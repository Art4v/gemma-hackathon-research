"""Mamba-1 selective scan: softplus(dt+bias) -> exp(dt*A) diagonal decay -> f32 recurrent state with input-dependent B/C -> D skip -> silu(z) gate."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mamba1_selective_scan',
    'batch': 4,
    'seq_len': 4096,
    'd_inner': 2048,
    'd_state': 16,
    'dt_init': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    l = CONFIG['seq_len']
    d = CONFIG['d_inner']
    n = CONFIG['d_state']
    k = jax.random.key(0)
    kx, kdt, kz, kb, kc = jax.random.split(k, 5)
    x = jax.random.normal(kx, (b, l, d), dtype)
    delta = jax.random.normal(kdt, (b, l, d), dtype)
    z = jax.random.normal(kz, (b, l, d), dtype)
    B = jax.random.normal(kb, (b, l, n), dtype)
    C = jax.random.normal(kc, (b, l, n), dtype)
    # Standard Mamba-1 S4D-real init: A = -[1..N] per channel.
    A_log = jnp.log(jnp.broadcast_to(jnp.arange(1, n + 1, dtype=jnp.float32), (d, n)))
    D = jnp.ones((d,), jnp.float32)
    # Inverse-softplus of dt_init so softplus(0 + dt_bias) == dt_init.
    dt0 = jnp.float32(CONFIG['dt_init'])
    dt_bias = jnp.full((d,), jnp.log(jnp.expm1(dt0)), jnp.float32)
    return x, delta, z, B, C, A_log, D, dt_bias


def workload(x, delta, z, B, C, A_log, D, dt_bias):
    out_dtype = x.dtype
    bsz, _, d = x.shape
    n = A_log.shape[-1]

    A = -jnp.exp(A_log.astype(jnp.float32))  # (D, N)
    xf = x.astype(jnp.float32)
    dt = jax.nn.softplus(delta.astype(jnp.float32) + dt_bias.astype(jnp.float32))  # (B, L, D)
    dtx = dt * xf

    # Time-major for the scan.
    dt_t = jnp.swapaxes(dt, 0, 1)
    dtx_t = jnp.swapaxes(dtx, 0, 1)
    B_t = jnp.swapaxes(B.astype(jnp.float32), 0, 1)
    C_t = jnp.swapaxes(C.astype(jnp.float32), 0, 1)

    def step(h, inp):
        dt_s, dtx_s, b_s, c_s = inp  # (B,D), (B,D), (B,N), (B,N)
        dA = jnp.exp(dt_s[:, :, None] * A[None, :, :])  # (B, D, N)
        h = dA * h + dtx_s[:, :, None] * b_s[:, None, :]
        y = jnp.einsum('bdn,bn->bd', h, c_s, precision=lax.Precision.HIGHEST)
        return h, y

    h0 = jnp.zeros((bsz, d, n), jnp.float32)
    _, ys = lax.scan(step, h0, (dt_t, dtx_t, B_t, C_t))
    y = jnp.swapaxes(ys, 0, 1)  # (B, L, D)

    y = y + xf * D.astype(jnp.float32)[None, None, :]
    y = y * jax.nn.silu(z.astype(jnp.float32))
    return y.astype(out_dtype)
