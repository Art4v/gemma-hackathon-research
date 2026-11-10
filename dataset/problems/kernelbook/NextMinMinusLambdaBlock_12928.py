"""[bf16 variant] Dual depthwise conv -> InstanceNorm+ReLU -> 2*min(l,r)-l -> LayerNorm -> Linear -> GELU -> Linear -> layer scale -> residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NextMinMinusLambdaBlock_12928',
    'batch': 8,
    'dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 7,
    'mlp_ratio': 4,
    'lambda_': 2.0,
    'in_eps': 1e-5,
    'ln_eps': 1e-6,
    'layer_scale_init_value': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    dim, k, hid = c['dim'], c['kernel_size'], c['mlp_ratio'] * c['dim']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], dim, c['height'], c['width']), dtype)
    wl = jax.random.normal(keys[1], (dim, 1, k, k), dtype) * 0.02
    bl = jax.random.normal(keys[2], (dim,), dtype) * 0.02
    wr = jax.random.normal(keys[3], (dim, 1, k, k), dtype) * 0.02
    br = jax.random.normal(keys[4], (dim,), dtype) * 0.02
    ln_w = jnp.ones((dim,), dtype)
    ln_b = jnp.zeros((dim,), dtype)
    w1 = jax.random.normal(keys[5], (dim, hid), dtype) * 0.02
    b1 = jax.random.normal(keys[6], (hid,), dtype) * 0.02
    w2 = jax.random.normal(keys[7], (hid, dim), dtype) * 0.02
    b2 = jax.random.normal(keys[8], (dim,), dtype) * 0.02
    gamma = jnp.full((dim,), CONFIG['layer_scale_init_value'], dtype)
    return (x, wl, bl, wr, br, ln_w, ln_b, w1, b1, w2, b2, gamma)


def _dwconv_in_relu(xh, w, b):
    pad = w.shape[-1] // 2
    y = lax.conv_general_dilated(
        xh, w, (1, 1), [(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        feature_group_count=xh.shape[-1])
    y = y + b
    m = jnp.mean(y, axis=(1, 2), keepdims=True)
    v = jnp.mean(jnp.square(y - m), axis=(1, 2), keepdims=True)
    return jax.nn.relu((y - m) * lax.rsqrt(v + CONFIG['in_eps']))


def workload(x, wl, bl, wr, br, ln_w, ln_b, w1, b1, w2, b2, gamma):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    left = _dwconv_in_relu(xh, wl, bl)
    right = _dwconv_in_relu(xh, wr, br)
    z = CONFIG['lambda_'] * jnp.minimum(left, right) - left
    m = jnp.mean(z, axis=-1, keepdims=True)
    v = jnp.mean(jnp.square(z - m), axis=-1, keepdims=True)
    z = (z - m) * lax.rsqrt(v + CONFIG['ln_eps']) * ln_w + ln_b
    z = jax.nn.gelu(z @ w1 + b1, approximate=False)
    z = (z @ w2 + b2) * gamma
    return x + jnp.transpose(z, (0, 3, 1, 2))
