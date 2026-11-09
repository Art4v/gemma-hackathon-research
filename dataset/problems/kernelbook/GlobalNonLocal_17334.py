"""GlobalNonLocal: 1x1 conv theta/phi/g -> softmax(theta^T phi) @ g -> 1x1 conv final -> + residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GlobalNonLocal_17334',
    'batch': 8,
    'in_channel': 512,
    'hidden_channel': 256,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    b, c, hid = CONFIG['batch'], CONFIG['in_channel'], CONFIG['hidden_channel']
    h, w = CONFIG['height'], CONFIG['width']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    w_theta = jax.random.normal(keys[1], (c, hid), dtype) * 0.02
    b_theta = jax.random.normal(keys[2], (hid,), dtype) * 0.02
    w_phi = jax.random.normal(keys[3], (c, hid), dtype) * 0.02
    b_phi = jax.random.normal(keys[4], (hid,), dtype) * 0.02
    w_g = jax.random.normal(keys[5], (c, hid), dtype) * 0.02
    b_g = jax.random.normal(keys[6], (hid,), dtype) * 0.02
    w_final = jax.random.normal(keys[7], (hid, c), dtype) * 0.02
    b_final = jax.random.normal(keys[8], (c,), dtype) * 0.02
    return x, w_theta, b_theta, w_phi, b_phi, w_g, b_g, w_final, b_final


def workload(x, w_theta, b_theta, w_phi, b_phi, w_g, b_g, w_final, b_final):
    b, c, h, w = x.shape
    # 1x1 convs are per-pixel matmuls; work in token layout (b, h*w, c).
    tokens = jnp.transpose(x.reshape(b, c, h * w), (0, 2, 1))
    theta = tokens @ w_theta + b_theta
    phi = tokens @ w_phi + b_phi
    g = tokens @ w_g + b_g
    attn = jax.nn.softmax(jnp.einsum('bnk,bmk->bnm', theta, phi), axis=-1)
    y = (attn @ g) @ w_final + b_final
    return jnp.transpose(y, (0, 2, 1)).reshape(b, c, h, w) + x
