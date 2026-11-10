"""ChannelAttentionGG: mean/max spatial pool, H-grid rolled weights, linear-relu-linear contraction, sigmoid, reshape."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ChannelAttentionGG_15323',
    'batch': 128,  # must equal N_in: torch broadcasting aligns the batch axis with N_in
    'channels': 8,  # must equal N_h_in: torch broadcasting aligns this axis with N_h_in
    'height': 256,
    'width': 256,
    'N_h': 4,
    'N_out': 128,
    'N_h_in': 8,
    'N_in': 128,
    'ratio': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_fc1, k_fc2 = jax.random.split(key, 3)
    n_out = CONFIG['N_out']
    n_in = CONFIG['N_in']
    n_r = n_in // CONFIG['ratio']
    n_h_in = CONFIG['N_h_in']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight_fc1 = jax.random.normal(k_fc1, (n_out, n_r, n_in, n_h_in), dtype) * 0.02
    weight_fc2 = jax.random.normal(k_fc2, (n_out, n_in, n_r, n_h_in), dtype) * 0.02
    return x, weight_fc1, weight_fc2


def _h_grid(w, n_h):
    return jnp.stack([jnp.roll(w, i, axis=-1) for i in range(n_h)], axis=1)


def _linear(x, w):
    x_reshaped = jnp.expand_dims(x, -4)
    w_reshaped = w[None, ..., None]
    return jnp.sum(x_reshaped * w_reshaped, axis=-3)


def workload(x, weight_fc1, weight_fc2):
    n_h = CONFIG['N_h']
    fc1 = _h_grid(weight_fc1, n_h)
    fc2 = _h_grid(weight_fc2, n_h)
    x_mean = jnp.mean(x, axis=(-2, -1))[..., None]
    x_max = jnp.max(x, axis=(-2, -1))[..., None]
    avg_out = _linear(jax.nn.relu(_linear(x_mean, fc1)), fc2)
    max_out = _linear(jax.nn.relu(_linear(x_max, fc1)), fc2)
    out = jax.nn.sigmoid(avg_out + max_out)
    return jnp.reshape(
        out,
        (x.shape[0], weight_fc1.shape[0], n_h, -1, weight_fc1.shape[-1], 1, 1),
    )
