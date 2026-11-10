"""DistillLoss: L2-normalize two feature tensors, pairwise cdist on each, same-cam off-diagonal mask, norm of masked difference over batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DistillLoss_18009',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'num_cams': 4,
    'cdist_eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch']
    c = CONFIG['channels']
    h = CONFIG['height']
    w = CONFIG['width']
    t_feat = jax.random.uniform(k1, (b, c, h, w), dtype)
    feat = jax.random.uniform(k2, (b, c, h, w), dtype)
    # The cams channel dim must equal batch for the mask to broadcast against the distances.
    cams = jax.random.randint(k3, (b, b, h, w), 0, CONFIG['num_cams'], jnp.int32)
    return t_feat, feat, cams


def _cdist(a, b):
    diff = jnp.expand_dims(a, 1) - jnp.expand_dims(b, 0)
    return jnp.sqrt(jnp.sum(diff ** 2, axis=2) + CONFIG['cdist_eps'])


def workload(t_feat, feat, cams):
    t_feat = t_feat / jnp.linalg.norm(t_feat, ord=2, axis=1, keepdims=True)
    t_dist = _cdist(t_feat, t_feat)
    feat = feat / jnp.linalg.norm(feat, ord=2, axis=1, keepdims=True)
    dist = _cdist(feat, feat)
    mask = (jnp.expand_dims(cams, 1) == jnp.expand_dims(cams, 0)).astype(feat.dtype)
    idx = jnp.arange(mask.shape[0])
    mask = mask.at[idx, idx].set(0)
    diff = (t_dist - dist) * mask
    return jnp.sqrt(jnp.sum(diff ** 2)) / feat.shape[0]
