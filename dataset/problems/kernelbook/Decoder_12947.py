"""Decoder: (mixture_w * est_mask) -> Linear basis (no bias) -> overlap-and-add with hop L//2."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Decoder_12947',
    'M': 8,
    'C': 2,
    'N': 256,
    'K': 8192,
    'L': 16,
    'audio_channels': 2,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    m, c, n, k = CONFIG['M'], CONFIG['C'], CONFIG['N'], CONFIG['K']
    ac, length = CONFIG['audio_channels'], CONFIG['L']
    mixture_w = jax.random.uniform(k1, (m, n, k), dtype)
    est_mask = jax.random.uniform(k2, (m, c, n, k), dtype)
    # Linear weight laid out (in=N, audio_channels, L) so workload reads sizes from shape.
    w = jax.random.normal(k3, (n, ac, length), dtype) * 0.02
    return mixture_w, est_mask, w


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _overlap_and_add(signal, frame_step):
    outer = signal.shape[:-2]
    frames, frame_length = signal.shape[-2:]
    sub_len = _gcd(frame_length, frame_step)
    sub_step = frame_step // sub_len
    subs_per_frame = frame_length // sub_len
    out_subs = (frame_step * (frames - 1) + frame_length) // sub_len
    x = signal.reshape(*outer, frames, subs_per_frame, sub_len)
    zero = jnp.array(0, signal.dtype)
    result = jnp.zeros((*outer, out_subs, sub_len), signal.dtype)
    # Subframe j of frame f lands at output subframe f*sub_step + j (same as torch index_add).
    for j in range(subs_per_frame):
        piece = x[..., :, j, :]
        high = out_subs - (j + (frames - 1) * sub_step + 1)
        pads = [(0, 0, 0)] * len(outer) + [(j, high, sub_step - 1), (0, 0, 0)]
        result = result + lax.pad(piece, zero, pads)
    return result.reshape(*outer, -1)


def workload(mixture_w, est_mask, w):
    length = w.shape[-1]
    source_w = jnp.expand_dims(mixture_w, 1) * est_mask  # [M, C, N, K]
    est_source = jnp.einsum('mcnk,nal->mcakl', source_w, w,
                            precision=lax.Precision.HIGHEST)  # [M, C, ac, K, L]
    return _overlap_and_add(est_source, length // 2)
