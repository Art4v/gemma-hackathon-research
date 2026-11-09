"""[bf16 variant] Multi-scale deformable attention: value/offset/weight projections -> softmax -> per-level bilinear grid sampling -> weighted sum -> output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'multiscale_deformable_attention',
    'batch': 8,
    'num_queries': 1024,
    'd_model': 256,
    'n_heads': 8,
    'n_levels': 4,
    'n_points': 4,
    'spatial_shapes': ((128, 128), (64, 64), (32, 32), (16, 16)),
    'seq_len': 128 * 128 + 64 * 64 + 32 * 32 + 16 * 16,
}


def create_inputs(dtype=jnp.bfloat16):
    b, nq, d = CONFIG['batch'], CONFIG['num_queries'], CONFIG['d_model']
    nh, nl, npt = CONFIG['n_heads'], CONFIG['n_levels'], CONFIG['n_points']
    s = CONFIG['seq_len']
    ks = jax.random.split(jax.random.key(0), 11)
    query = jax.random.normal(ks[0], (b, nq, d), dtype)
    reference_points = jax.random.uniform(ks[1], (b, nq, nl, 2), dtype)
    value_input = jax.random.normal(ks[2], (b, s, d), dtype)
    n_off = nh * nl * npt * 2
    n_aw = nh * nl * npt
    w_off = jax.random.normal(ks[3], (d, n_off), dtype) * 0.02
    b_off = jax.random.normal(ks[4], (n_off,), dtype) * 0.02
    w_aw = jax.random.normal(ks[5], (d, n_aw), dtype) * 0.02
    b_aw = jax.random.normal(ks[6], (n_aw,), dtype) * 0.02
    w_v = jax.random.normal(ks[7], (d, d), dtype) * 0.02
    b_v = jax.random.normal(ks[8], (d,), dtype) * 0.02
    w_out = jax.random.normal(ks[9], (d, d), dtype) * 0.02
    b_out = jax.random.normal(ks[10], (d,), dtype) * 0.02
    return (query, reference_points, value_input,
            w_off, b_off, w_aw, b_aw, w_v, b_v, w_out, b_out)


def _bilinear_sample(value_l, loc, height, width):
    """grid_sample(bilinear, zeros padding, align_corners=False).

    value_l: (b, nh, H*W, hd); loc: (b, nh, nq, P, 2) as (x, y) in [0, 1].
    Returns (b, nh, nq, P, hd).
    """
    b, nh, _, hd = value_l.shape
    x = loc[..., 0] * width - 0.5
    y = loc[..., 1] * height - 0.5
    x0f = jnp.floor(x)
    y0f = jnp.floor(y)
    fx = x - x0f
    fy = y - y0f
    x0 = x0f.astype(jnp.int32)
    y0 = y0f.astype(jnp.int32)
    out = jnp.zeros(loc.shape[:-1] + (hd,), value_l.dtype)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            xi = x0 + dx
            yi = y0 + dy
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            idx = jnp.where(valid, yi * width + xi, 0)
            g = jnp.take_along_axis(value_l, idx.reshape(b, nh, -1, 1), axis=2)
            w = jnp.where(valid, wx * wy, 0.0)
            out = out + g.reshape(idx.shape + (hd,)) * w[..., None]
    return out


def workload(query, reference_points, value_input,
             w_off, b_off, w_aw, b_aw, w_v, b_v, w_out, b_out):
    nh, nl, npt = CONFIG['n_heads'], CONFIG['n_levels'], CONFIG['n_points']
    shapes = CONFIG['spatial_shapes']
    b, nq, d = query.shape
    hd = d // nh

    value = (value_input @ w_v + b_v).reshape(b, -1, nh, hd).transpose(0, 2, 1, 3)
    off = (query @ w_off + b_off).reshape(b, nq, nh, nl, npt, 2)
    aw = (query @ w_aw + b_aw).reshape(b, nq, nh, nl * npt)
    aw = jax.nn.softmax(aw, axis=-1).reshape(b, nq, nh, nl, npt)
    norm = jnp.array([[w, h] for h, w in shapes], dtype=query.dtype)
    loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]

    loc = loc.transpose(0, 2, 1, 3, 4, 5)  # (b, nh, nq, nl, P, 2)
    aw = aw.transpose(0, 2, 1, 3, 4)       # (b, nh, nq, nl, P)

    out = jnp.zeros((b, nh, nq, hd), query.dtype)
    start = 0
    for lvl, (h, w) in enumerate(shapes):
        v_l = value[:, :, start:start + h * w]
        start += h * w
        s = _bilinear_sample(v_l, loc[:, :, :, lvl], h, w)
        out = out + jnp.sum(s * aw[:, :, :, lvl, :, None], axis=3)

    out = out.transpose(0, 2, 1, 3).reshape(b, nq, d)
    return out @ w_out + b_out
