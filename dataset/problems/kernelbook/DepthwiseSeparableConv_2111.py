"""[bf16 variant] Depthwise Conv1d (groups=C, pad K//2, no bias) -> pointwise 1x1 Conv1d with bias -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DepthwiseSeparableConv_2111',
    'batch': 16,
    'in_channels': 512,
    'out_channels': 512,
    'length': 8192,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_dw, k_pw, k_b = jax.random.split(jax.random.key(0), 4)
    n, c_in, c_out = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    length, k = CONFIG['length'], CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (n, c_in, length), dtype)
    dw_w = jax.random.normal(k_dw, (c_in, 1, k), dtype) * 0.02  # PyTorch (C, 1, K)
    pw_w = jax.random.normal(k_pw, (c_in, c_out), dtype) * 0.02  # (in, out)
    pw_b = jax.random.normal(k_b, (c_out,), dtype) * 0.02
    return x, dw_w, pw_w, pw_b


def workload(x, dw_w, pw_w, pw_b):
    # PyTorch Conv1d accepts unbatched (C, L) input; mirror that.
    is_unbatched = x.ndim == 2
    if is_unbatched:
        x = x[None]
    c_in = x.shape[1]
    k = dw_w.shape[-1]
    x_nwc = jnp.transpose(x, (0, 2, 1))
    y = lax.conv_general_dilated(
        x_nwc, dw_w,
        window_strides=(1,),
        padding=[(k // 2, k // 2)],
        dimension_numbers=('NWC', 'OIW', 'NWC'),
        feature_group_count=c_in,
    )
    y = jax.nn.relu(y @ pw_w + pw_b)
    y = jnp.transpose(y, (0, 2, 1))
    if is_unbatched:
        y = y[0]
    return y
