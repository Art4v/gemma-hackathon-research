"""[bf16 variant] Replicate-padded dilated Conv1d -> ReLU -> 1x1 Conv1d -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DilatedResidualLayer_8122',
    'batch': 16,
    'channels': 256,
    'length': 16384,
    'kernel_size': 3,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_wd, k_bd, k_wo, k_bo = jax.random.split(jax.random.key(0), 5)
    n, c, l, k = CONFIG['batch'], CONFIG['channels'], CONFIG['length'], CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (n, c, l), dtype)
    w_dil = jax.random.normal(k_wd, (k, c, c), dtype) * 0.02
    b_dil = jax.random.normal(k_bd, (c,), dtype) * 0.02
    w_out = jax.random.normal(k_wo, (c, c), dtype) * 0.02
    b_out = jax.random.normal(k_bo, (c,), dtype) * 0.02
    return x, w_dil, b_dil, w_out, b_out


def workload(x, w_dil, b_dil, w_out, b_out):
    d = CONFIG['dilation']
    # KernelBench feeds an unbatched (C, L) tensor; accept both layouts.
    unbatched = x.ndim == 2
    if unbatched:
        x = x[None]
    h = jnp.transpose(x, (0, 2, 1))  # NCL -> NLC
    hp = jnp.pad(h, ((0, 0), (d, d), (0, 0)), mode='edge')  # padding_mode='replicate'
    y = lax.conv_general_dilated(
        hp, w_dil, window_strides=(1,), padding='VALID', rhs_dilation=(d,),
        dimension_numbers=('NWC', 'WIO', 'NWC'), precision=lax.Precision.HIGHEST)
    y = jax.nn.relu(y + b_dil)
    y = jnp.matmul(y, w_out, precision=lax.Precision.HIGHEST) + b_out
    out = jnp.transpose(h + y, (0, 2, 1))  # NLC -> NCL
    return out[0] if unbatched else out
