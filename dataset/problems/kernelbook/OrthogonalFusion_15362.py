"""OrthogonalFusion: per-batch projection of local features onto global vector, subtract, concat with broadcast global."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OrthogonalFusion_15362',
    'batch': 16,
    'channels': 2048,
    'height': 32,
    'width': 32,
}


def create_inputs(dtype=jnp.float32):
    k_local, k_global = jax.random.split(jax.random.key(0))
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    local_feat = jax.random.uniform(k_local, (b, c, h, w), dtype)
    global_feat = jax.random.uniform(k_global, (b, c), dtype)
    return local_feat, global_feat


def workload(local_feat, global_feat):
    b, c = global_feat.shape
    norm = jnp.sqrt(jnp.sum(global_feat * global_feat, axis=1))
    flat = local_feat.reshape(b, c, -1)
    # bmm(g.unsqueeze(1), flat) -> (B, L)
    proj = jnp.einsum('bc,bcl->bl', global_feat, flat, precision=lax.Precision.HIGHEST)
    # bmm(g.unsqueeze(2), proj) -> (B, C, L), viewed as local_feat's shape
    proj = jnp.einsum('bc,bl->bcl', global_feat, proj,
                      precision=lax.Precision.HIGHEST).reshape(local_feat.shape)
    # ponytail: literal torch broadcasting (also covers the 3D B==C get_inputs case)
    proj = proj / (norm * norm).reshape(-1, 1, 1, 1)
    orth = local_feat - proj
    g = jnp.broadcast_to(global_feat[:, :, None, None], orth.shape)
    return jnp.concatenate([g, orth], axis=1)
