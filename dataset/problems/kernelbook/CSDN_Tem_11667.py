"""[bf16 variant] Depthwise 3x3 conv (bias) -> pointwise 1x1 conv (bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CSDN_Tem_11667',
    'batch': 8,
    'in_ch': 128,
    'out_ch': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.key(0)
    kx, kdw, kdb, kpw, kpb = jax.random.split(k, 5)
    x = jax.random.uniform(kx, (c['batch'], c['in_ch'], c['height'], c['width']), dtype)
    dw = jax.random.normal(kdw, (c['kernel_size'], c['kernel_size'], 1, c['in_ch']), dtype) * 0.02
    db = jax.random.normal(kdb, (c['in_ch'],), dtype) * 0.02
    pw = jax.random.normal(kpw, (c['in_ch'], c['out_ch']), dtype) * 0.02
    pb = jax.random.normal(kpb, (c['out_ch'],), dtype) * 0.02
    return x, dw, db, pw, pb


def workload(x, dw, db, pw, pb):
    p = CONFIG['padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = lax.conv_general_dilated(
        h, dw, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=x.shape[1],
        precision=lax.Precision.HIGHEST,
    ) + db
    h = jnp.einsum('nhwc,co->nhwo', h, pw, precision=lax.Precision.HIGHEST) + pb
    return jnp.transpose(h, (0, 3, 1, 2))
