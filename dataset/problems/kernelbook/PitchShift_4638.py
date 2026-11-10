"""[bf16 variant] PitchShift: bilinear stretch of the mel axis (align_corners=False), crop/zero-pad to mel size, insert channel dim."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PitchShift_4638',
    'batch': 128,
    'mel_size': 1024,
    'time_steps': 512,
    'shift': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['mel_size'], CONFIG['time_steps']), dtype
    )
    return (x,)


def workload(x):
    if x.ndim == 2:
        x = x[None]
    x = jnp.squeeze(x)
    mel_size = x.shape[1]
    shift_scale = (mel_size + CONFIG['shift']) / mel_size
    out_size = int(mel_size * shift_scale)

    # src = (d + 0.5) * mel_size / out_size - 0.5, as an exact rational num / den.
    dst = jnp.arange(out_size, dtype=jnp.int32)
    den = 2 * out_size
    num = jnp.maximum((2 * dst + 1) * mel_size - out_size, 0)
    lower = num // den
    upper = jnp.minimum(lower + 1, mel_size - 1)
    frac = (num % den).astype(x.dtype) / jnp.asarray(den, x.dtype)
    frac = frac.reshape((1, out_size, 1))

    x = (1 - frac) * jnp.take(x, lower, axis=1) + frac * jnp.take(x, upper, axis=1)

    x = x[:, :mel_size]
    if x.shape[1] < mel_size:
        pad_size = mel_size - x.shape[1]
        x = jnp.concatenate(
            [x, jnp.zeros((x.shape[0], pad_size, x.shape[2]), x.dtype)], axis=1
        )
    x = jnp.squeeze(x)
    return jnp.expand_dims(x, 1)
