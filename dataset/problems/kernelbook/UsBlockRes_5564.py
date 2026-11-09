"""[bf16 variant] UsBlockRes: ConvTranspose3d(k=2,s=2) -> add skip -> Conv3d(k=3,p=1) -> ReLU (unbatched CDHW)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UsBlockRes_5564',
    'in_channels': 128,
    'out_channels': 128,
    'in_spatial': 32,
    'up_kernel': 2,
    'up_stride': 2,
    'conv_kernel': 3,
    'conv_padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    cin, cout = CONFIG['in_channels'], CONFIG['out_channels']
    s = CONFIG['in_spatial']
    ku, kc = CONFIG['up_kernel'], CONFIG['conv_kernel']
    so = s * CONFIG['up_stride']
    k = jax.random.split(jax.random.key(0), 6)
    before_pool = jax.random.uniform(k[0], (cout, so, so, so), dtype)
    x = jax.random.uniform(k[1], (cin, s, s, s), dtype)
    w_up = jax.random.normal(k[2], (cin, cout, ku, ku, ku), dtype) * 0.02
    b_up = jax.random.normal(k[3], (cout,), dtype) * 0.02
    w_conv = jax.random.normal(k[4], (cout, cout, kc, kc, kc), dtype) * 0.02
    b_conv = jax.random.normal(k[5], (cout,), dtype) * 0.02
    return before_pool, x, w_up, b_up, w_conv, b_conv


def workload(before_pool, x, w_up, b_up, w_conv, b_conv):
    _, d, h, w = x.shape
    cout, ku = w_up.shape[1], w_up.shape[2]
    # Transposed conv with kernel == stride has no overlap: each input voxel
    # writes its own ku^3 output block, so one einsum + reshape is exact.
    up = jnp.einsum('cdhw,coijk->odihjwk', x, w_up)
    up = up.reshape(cout, d * ku, h * ku, w * ku) + b_up[:, None, None, None]
    y = up + before_pool
    lhs = jnp.transpose(y, (1, 2, 3, 0))[None]           # (1, D, H, W, C)
    rhs = jnp.transpose(w_conv, (2, 3, 4, 1, 0))         # DHWIO
    p = CONFIG['conv_padding']
    out = lax.conv_general_dilated(
        lhs, rhs, window_strides=(1, 1, 1), padding=((p, p),) * 3,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    out = out[0] + b_conv
    out = jnp.maximum(out, 0.0)
    return jnp.transpose(out, (3, 0, 1, 2))              # (C, D, H, W)
