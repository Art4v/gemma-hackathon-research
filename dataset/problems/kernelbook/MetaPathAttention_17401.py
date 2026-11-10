"""MetaPathAttention: concat(user, item, metapath_i) -> Linear -> ReLU -> Linear -> ReLU -> softmax over metapaths -> attention-weighted metapath sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MetaPathAttention_17401',
    'batch': 1024,
    'latent_dim': 512,
    'metapath_type_num': 128,
    'att_size': 512,
    'concat_factor': 3,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['batch']
    d = CONFIG['latent_dim']
    m = CONFIG['metapath_type_num']
    a = CONFIG['att_size']
    in_dim = CONFIG['concat_factor'] * d
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 7)
    user_latent = jax.random.uniform(keys[0], (n, d), dtype)
    item_latent = jax.random.uniform(keys[1], (n, d), dtype)
    metapath_latent = jax.random.uniform(keys[2], (n, m, d), dtype)
    w1 = jax.random.normal(keys[3], (in_dim, a), dtype) * scale
    b1 = jax.random.normal(keys[4], (a,), dtype) * scale
    w2 = jax.random.normal(keys[5], (a, 1), dtype) * scale
    b2 = jax.random.normal(keys[6], (1,), dtype) * scale
    return user_latent, item_latent, metapath_latent, w1, b1, w2, b2


def workload(user_latent, item_latent, metapath_latent, w1, b1, w2, b2):
    n, m = metapath_latent.shape[:2]
    user = jnp.broadcast_to(user_latent[:, None, :], (n, m, user_latent.shape[-1]))
    item = jnp.broadcast_to(item_latent[:, None, :], (n, m, item_latent.shape[-1]))
    inputs = jnp.concatenate((user, item, metapath_latent), axis=-1)
    hidden = jax.nn.relu(inputs @ w1 + b1)
    scores = jax.nn.relu(hidden @ w2 + b2)
    atten = jax.nn.softmax(jnp.squeeze(scores, axis=-1), axis=1)
    return jnp.sum(metapath_latent * atten[..., None], axis=1)
