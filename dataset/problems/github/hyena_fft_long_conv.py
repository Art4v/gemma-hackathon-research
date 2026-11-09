"""[bf16 variant] Hyena order-2 operator: in-proj -> causal depthwise short conv -> gated FFT long conv (+D skip) -> gate -> out-proj."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'hyena_fft_long_conv',
    'batch': 8,
    'seq_len': 4096,
    'd_model': 1024,
    'l_max': 4096,
    'order': 2,
    'short_filter_order': 3,
    'filter_decay': 8.0,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    b, l, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    order, k, l_max = CONFIG['order'], CONFIG['short_filter_order'], CONFIG['l_max']
    inner = d * (order + 1)
    keys = jax.random.split(jax.random.key(0), 10)
    u = jax.random.normal(keys[0], (b, l, d), dtype)
    w_in = jax.random.normal(keys[1], (d, inner), dtype) * 0.02
    b_in = jax.random.normal(keys[2], (inner,), dtype) * 0.02
    w_short = jax.random.normal(keys[3], (k, 1, inner), dtype) * 0.02
    b_short = jax.random.normal(keys[4], (inner,), dtype) * 0.02
    decay = jnp.exp(-jnp.linspace(0.0, CONFIG['filter_decay'], l_max)).astype(dtype)
    long_filter = jax.random.normal(keys[5], (order, d, l_max), dtype) * 0.02 * decay[None, None, :]
    bias = jax.random.normal(keys[6], (order, d), dtype) * 0.02
    w_out = jax.random.normal(keys[7], (d, d), dtype) * 0.02
    b_out = jax.random.normal(keys[8], (d,), dtype) * 0.02
    return u, w_in, b_in, w_short, b_short, long_filter, bias, w_out, b_out


def _fftconv(u, k, skip):
    # u: [B, L, C], k: [C, L], skip: [C]. Causal linear conv via zero-padded FFT, torch normalization.
    seqlen = u.shape[1]
    n = 2 * seqlen
    k_f = jnp.fft.rfft(k, n=n, axis=-1).T / n
    u_f = jnp.fft.rfft(u, n=n, axis=1)
    y = jnp.fft.irfft(u_f * k_f[None], n=n, axis=1, norm='forward')[:, :seqlen]
    return (y + u * skip).astype(u.dtype)


def workload(u, w_in, b_in, w_short, b_short, long_filter, bias, w_out, b_out):
    l = u.shape[1]
    ksize = w_short.shape[0]
    inner = w_in.shape[1]
    z = jnp.matmul(u, w_in, precision=_HI) + b_in  # [B, L, inner], NWC
    uc = lax.conv_general_dilated(
        z, w_short, window_strides=(1,), padding=[(ksize - 1, 0)],
        dimension_numbers=('NWC', 'WIO', 'NWC'), feature_group_count=inner,
        precision=_HI) + b_short
    *x, v = jnp.split(uc, CONFIG['order'] + 1, axis=-1)
    for o, x_i in enumerate(reversed(x[1:])):
        v = v * x_i
        v = _fftconv(v, long_filter[o, :, :l], bias[o])
    y = v * x[0]
    return jnp.matmul(y, w_out, precision=_HI) + b_out
