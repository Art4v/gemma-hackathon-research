"""PitchShift: bilinear resize along the mel axis by (M+shift)/M, crop/zero-pad back to M, unsqueeze(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PitchShift_16402',
    'batch': 16,
    'mel_size': 1024,
    'time': 2048,
    'shift': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['mel_size'], CONFIG['time']), dtype)
    return (x,)


def workload(x):
    m = x.shape[1]
    # PyTorch recompute_scale_factor=True: out = floor(in * scale), coords use in/out.
    out_h = int(m * ((m + CONFIG['shift']) / m))
    keep = min(out_h, m)
    scale = jnp.float32(m) / jnp.float32(out_h)
    dst = jnp.arange(keep, dtype=jnp.float32)
    src = jnp.maximum((dst + 0.5) * scale - 0.5, 0.0)
    i0 = jnp.floor(src).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, m - 1)
    lam = (src - i0.astype(jnp.float32)).astype(x.dtype)[None, :, None]
    # Width scale is exactly 1.0 with align_corners=False, so that axis is an identity.
    y = (1.0 - lam) * x[:, i0, :] + lam * x[:, i1, :]
    y = jnp.pad(y, ((0, 0), (0, m - keep), (0, 0)))
    return y[:, None, :, :]
