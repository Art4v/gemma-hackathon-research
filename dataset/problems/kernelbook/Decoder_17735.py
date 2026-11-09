"""[bf16 variant] Decoder_17735: concat(z, c) -> Linear -> ReLU -> Linear -> Softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Decoder_17735',
    'batch': 8,
    'height': 128,
    'width': 128,
    'latent_size': 256,
    'nlabels': 128,
    'layer_sizes': [512, 256],
    'softmax_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b, h, w = CONFIG['batch'], CONFIG['height'], CONFIG['width']
    sizes = CONFIG['layer_sizes']
    in_sizes = [CONFIG['latent_size'] + CONFIG['nlabels']] + sizes[:-1]
    keys = jax.random.split(jax.random.key(0), 2 + len(sizes))
    z = jax.random.uniform(keys[0], (b, h, w, CONFIG['latent_size']), dtype)
    c = jax.random.uniform(keys[1], (b, h, w, CONFIG['nlabels']), dtype)
    params = []
    for k, n_in, n_out in zip(keys[2:], in_sizes, sizes):
        params.append(jax.random.normal(k, (n_in, n_out), dtype) * 0.02)
        params.append(jnp.zeros((n_out,), dtype))
    return (z, c, *params)


def workload(z, c, *params):
    x = jnp.concatenate((z, c), axis=-1)
    n_layers = len(params) // 2
    for i in range(n_layers):
        x = x @ params[2 * i] + params[2 * i + 1]
        if i + 1 < n_layers:
            x = jax.nn.relu(x)
    # ponytail: matches torch Softmax(dim=1) on a 4D tensor, i.e. over H, not features.
    return jax.nn.softmax(x, axis=CONFIG['softmax_dim'])
