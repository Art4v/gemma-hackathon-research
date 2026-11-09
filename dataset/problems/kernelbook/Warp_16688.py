"""Warp: time-axis grid_sample (bilinear, reflection padding, align_corners=True) with grid = (linspace over assets, tform over lookback)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Warp_16688',
    'n_samples': 16,
    'n_channels': 32,
    'lookback': 512,
    'n_assets': 256,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_t = jax.random.split(jax.random.key(0))
    n = CONFIG['n_samples']
    c = CONFIG['n_channels']
    lb = CONFIG['lookback']
    na = CONFIG['n_assets']
    x = jax.random.uniform(k_x, (n, c, lb, na), dtype)
    tform = jax.random.uniform(k_t, (n, lb), dtype)
    return x, tform


def _source_index(coord, size):
    """PyTorch grid_sampler_compute_source_index, align_corners=True, reflection padding."""
    ix = ((coord + 1.0) / 2.0) * (size - 1)
    if size <= 1:
        # reflect_coordinates returns 0 when twice_low == twice_high
        return jnp.zeros_like(ix)
    span = jnp.asarray(size - 1, ix.dtype)  # (2*(size-1) - 0) / 2, min = 0
    a = jnp.abs(ix)
    extra = jnp.fmod(a, span)
    flips = jnp.floor(a / span)
    reflected = jnp.where(jnp.fmod(flips, 2.0) == 0.0, extra, span - extra)
    return jnp.clip(reflected, 0.0, size - 1)


def _linear_taps(coord, size):
    """Two-tap linear interpolation indices/weights with PyTorch's out-of-bounds zeroing."""
    i0f = jnp.floor(coord)
    frac = coord - i0f
    i0 = i0f.astype(jnp.int32)
    i1 = i0 + 1
    w0 = 1.0 - frac
    w1 = jnp.where(i1 <= size - 1, frac, 0.0)
    i0 = jnp.clip(i0, 0, size - 1)
    i1 = jnp.clip(i1, 0, size - 1)
    return i0, i1, w0, w1


def workload(x, tform):
    if tform.ndim != 2:
        raise ValueError('This port implements the 2-D tform (n_samples, lookback) path.')
    _, _, h_in, w_in = x.shape
    out_dtype = x.dtype
    xf = x.astype(jnp.float32)

    # grid[..., 1] = tform[n, h_out] (shared across all assets); grid[..., 0] = linspace(-1, 1, n_assets)
    gy = tform.astype(jnp.float32)                              # (N, H_out)
    gx = jnp.linspace(-1.0, 1.0, w_in, dtype=jnp.float32)      # (W_out,)

    iy = _source_index(gy, h_in)
    ix = _source_index(gx, w_in)
    y0, y1, wy0, wy1 = _linear_taps(iy, h_in)                   # (N, H_out)
    x0, x1, wx0, wx1 = _linear_taps(ix, w_in)                   # (W_out,)

    # Separable bilinear: blend along H (per-sample row gather), then along W.
    take_rows = jax.vmap(lambda xn, idx: xn[:, idx, :])          # (C,H,W),(H_out,) -> (C,H_out,W)
    rows = (wy0[:, None, :, None] * take_rows(xf, y0)
            + wy1[:, None, :, None] * take_rows(xf, y1))         # (N, C, H_out, W)
    out = rows[..., x0] * wx0 + rows[..., x1] * wx1              # (N, C, H_out, W_out)
    return out.astype(out_dtype)
