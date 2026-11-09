"""[bf16 variant] Capsule dynamic routing: u_hat einsum, then iterated softmax(dim=in_caps), weighted sum, squash and agreement update."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'capsule_dynamic_routing',
    'batch': 128,
    'in_caps': 4096,
    'out_caps': 32,
    'in_d': 8,
    'out_d': 16,
    'iterations': 4,
    'epsilon': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    k_u, k_w = jax.random.split(jax.random.key(0))
    u = jax.random.uniform(k_u, (CONFIG['batch'], CONFIG['in_caps'], CONFIG['in_d']), dtype)
    weight = jax.random.normal(
        k_w, (CONFIG['in_caps'], CONFIG['out_caps'], CONFIG['in_d'], CONFIG['out_d']), dtype
    ) * 0.02
    return u, weight


def _squash(s):
    s2 = jnp.sum(s * s, axis=-1, keepdims=True)
    return s2 / (1 + s2) * (s / jnp.sqrt(s2 + CONFIG['epsilon']))


def workload(u, weight):
    u_hat = jnp.einsum('ijnm,bin->bijm', weight, u)
    b = jnp.zeros((u.shape[0], weight.shape[0], weight.shape[1]), u.dtype)
    v = None
    for _ in range(CONFIG['iterations']):
        c = jax.nn.softmax(b, axis=1)
        s = jnp.einsum('bij,bijm->bjm', c, u_hat)
        v = _squash(s)
        b = b + jnp.einsum('bjm,bijm->bij', v, u_hat)
    return v
