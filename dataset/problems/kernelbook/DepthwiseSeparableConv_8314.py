"""[bf16 variant] Depthwise Conv1d (groups=C, padding=k//2) + bias -> pointwise Conv1d (1x1) + bias -> ReLU, on (N, L, D) input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DepthwiseSeparableConv_8314',
    'batch': 16,
    'seq_len': 2048,
    'in_ch': 1024,
    'out_ch': 1024,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_dw, k_dwb, k_pw, k_pwb = jax.random.split(jax.random.key(0), 5)
    n, l = CONFIG['batch'], CONFIG['seq_len']
    c, o, k = CONFIG['in_ch'], CONFIG['out_ch'], CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (n, l, c), dtype)
    w_dw = jax.random.normal(k_dw, (k, 1, c), dtype) * 0.02
    b_dw = jax.random.normal(k_dwb, (c,), dtype) * 0.02
    w_pw = jax.random.normal(k_pw, (c, o), dtype) * 0.02
    b_pw = jax.random.normal(k_pwb, (o,), dtype) * 0.02
    return x, w_dw, b_dw, w_pw, b_pw


def workload(x, w_dw, b_dw, w_pw, b_pw):
    # x is (N, L, D): already channels-last (NWC), so torch's transpose pair is a no-op here.
    pad = CONFIG['kernel_size'] // 2
    h = lax.conv_general_dilated(
        x, w_dw,
        window_strides=(1,),
        padding=((pad, pad),),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=x.shape[-1],
        precision=lax.Precision.HIGHEST,
    ) + b_dw
    out = jnp.matmul(h, w_pw, precision=lax.Precision.HIGHEST) + b_pw
    return jax.nn.relu(out)
