"""Decoder_6333: reshape -> split flattened (levels*eccentricity) axis in half -> regroup -> permute -> concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Decoder_6333',
    'batch': 8,
    'n_levels': 4,
    'n_color': 4,
    'n_eccentricity': 8,
    'n_azimuth': 128,
    'n_theta': 8,
    'n_phase': 32,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    shape = (c['batch'], c['n_color'], c['n_theta'], c['n_phase'],
             c['n_levels'], c['n_eccentricity'], c['n_azimuth'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (x,)


def workload(x):
    # x: (B, C, T, P, L, E, A); the 7D shape carries every size.
    b, c, t, p, l, e, a = x.shape
    flat = x.reshape(b, c, t, p, l * e, a)
    lim = l * e // 2
    # Split on the flattened L*E axis, then regroup each half as (L, E//2), as the torch view does.
    halves = (flat[:, :, :, :, :lim], flat[:, :, :, :, lim:])
    parts = [
        jnp.transpose(h.reshape(b, c, t, p, l, e // 2, a), (0, 4, 1, 5, 6, 2, 3))
        for h in halves
    ]
    return jnp.concatenate(parts, axis=3)
