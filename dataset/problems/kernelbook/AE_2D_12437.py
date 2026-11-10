"""[bf16 variant] AE_2D autoencoder: (Linear + tanh) x3, Linear, tanh, (Linear + tanh) x3, Linear applied on the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AE_2D_12437',
    'batch': 8,
    'height': 128,
    'width': 128,
    'n_features': 256,
    'hidden_sizes': (1280, 640, 384, 128),
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    n_features = CONFIG['n_features']
    hidden = CONFIG['hidden_sizes']
    scale = CONFIG['weight_scale']
    sizes = (n_features, *hidden, *hidden[-2::-1], n_features)
    num_layers = len(sizes) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * num_layers)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], n_features),
        dtype,
    )
    params = []
    for i in range(num_layers):
        w = jax.random.normal(keys[1 + 2 * i], (sizes[i], sizes[i + 1]), dtype) * scale
        b = jax.random.normal(keys[2 + 2 * i], (sizes[i + 1],), dtype) * scale
        params.extend((w, b))
    return (x, *params)


def workload(*inputs):
    (x,
     en1_w, en1_b, en2_w, en2_b, en3_w, en3_b, en4_w, en4_b,
     de1_w, de1_b, de2_w, de2_b, de3_w, de3_b, de4_w, de4_b) = inputs
    h = jnp.tanh(x @ en1_w + en1_b)
    h = jnp.tanh(h @ en2_w + en2_b)
    h = jnp.tanh(h @ en3_w + en3_b)
    z = h @ en4_w + en4_b
    h = jnp.tanh(jnp.tanh(z) @ de1_w + de1_b)
    h = jnp.tanh(h @ de2_w + de2_b)
    h = jnp.tanh(h @ de3_w + de3_b)
    return h @ de4_w + de4_b
