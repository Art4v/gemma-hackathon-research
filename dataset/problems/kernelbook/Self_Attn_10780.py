"""[bf16 variant] Self-attention: 1x1 Conv2d query/key/value -> flatten -> bmm -> softmax -> bmm -> gamma-scaled residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Self_Attn_10780',
    'batch': 8,
    'in_dim': 512,
    'proj_dim': 128,
    'width': 64,
    'height': 64,
    'kernel_size': 1,
    'stride': 1,
    'weight_scale': 0.02,
    'gamma_init': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    in_dim = CONFIG['in_dim']
    proj_dim = CONFIG['proj_dim']
    ks = CONFIG['kernel_size']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], in_dim, CONFIG['width'], CONFIG['height']), dtype
    )
    query_w = jax.random.normal(keys[1], (proj_dim, in_dim, ks, ks), dtype) * scale
    query_b = jax.random.normal(keys[2], (proj_dim,), dtype) * scale
    key_w = jax.random.normal(keys[3], (proj_dim, in_dim, ks, ks), dtype) * scale
    key_b = jax.random.normal(keys[4], (proj_dim,), dtype) * scale
    value_w = jax.random.normal(keys[5], (in_dim, in_dim, ks, ks), dtype) * scale
    value_b = jax.random.normal(keys[6], (in_dim,), dtype) * scale
    gamma = jnp.ones((1,), dtype) * CONFIG['gamma_init']
    return x, query_w, query_b, key_w, key_b, value_w, value_b, gamma


def _conv_flat(x_nhwc, w, b, batch, n):
    stride = CONFIG['stride']
    y = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=(stride, stride),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + b
    y = jnp.transpose(y, (0, 3, 1, 2))
    return y.reshape(batch, -1, n)


def workload(x, query_w, query_b, key_w, key_b, value_w, value_b, gamma):
    batch, channels, width, height = x.shape
    n = width * height
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    proj_query = jnp.transpose(_conv_flat(x_nhwc, query_w, query_b, batch, n), (0, 2, 1))
    proj_key = _conv_flat(x_nhwc, key_w, key_b, batch, n)
    energy = jnp.matmul(proj_query, proj_key)
    attention = jax.nn.softmax(energy, axis=-1)
    proj_value = _conv_flat(x_nhwc, value_w, value_b, batch, n)
    out = jnp.matmul(proj_value, attention)
    out = out.reshape(batch, channels, width, height)
    return gamma * out + x
