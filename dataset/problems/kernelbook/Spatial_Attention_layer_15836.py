"""[bf16 variant] Spatial attention: bilinear form (x @ W1 @ W2) @ (W3 @ x)^T, plus bias, sigmoid, Vs matmul, softmax over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Spatial_Attention_layer_15836',
    'batch_size': 8,
    'num_of_vertices': 512,
    'in_channels': 128,
    'num_of_timesteps': 128,
    'weight_scale': 0.02,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch_size']
    n = CONFIG['num_of_vertices']
    f = CONFIG['in_channels']
    t = CONFIG['num_of_timesteps']
    scale = CONFIG['weight_scale']
    k_x, k_w1, k_w2, k_w3, k_bs, k_vs = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(k_x, (b, n, f, t), dtype)
    w1 = jax.random.normal(k_w1, (t,), dtype) * scale
    w2 = jax.random.normal(k_w2, (f, t), dtype) * scale
    w3 = jax.random.normal(k_w3, (f,), dtype) * scale
    bs = jax.random.normal(k_bs, (1, n, n), dtype) * scale
    vs = jax.random.normal(k_vs, (n, n), dtype) * scale
    return x, w1, w2, w3, bs, vs


def workload(x, w1, w2, w3, bs, vs):
    lhs = jnp.matmul(jnp.matmul(x, w1), w2)
    rhs = jnp.swapaxes(jnp.matmul(w3, x), -1, -2)
    product = jnp.matmul(lhs, rhs)
    s = jnp.matmul(vs, jax.nn.sigmoid(product + bs))
    return jax.nn.softmax(s, axis=CONFIG['softmax_axis'])
