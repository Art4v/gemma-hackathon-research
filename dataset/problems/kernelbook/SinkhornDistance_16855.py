"""[bf16 variant] SinkhornDistance: 100 log-domain Sinkhorn iterations (logsumexp row/col updates) -> pi = exp(M), cost = sum(pi * C)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SinkhornDistance_16855',
    'batch': 8,
    'p1': 2048,
    'p2': 2048,
    'eps': 1e-3,
    'max_iter': 100,
    'log_offset': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    k_mu, k_nu, k_c = jax.random.split(jax.random.key(0), 3)
    b, p1, p2 = CONFIG['batch'], CONFIG['p1'], CONFIG['p2']
    mu = jax.random.uniform(k_mu, (b, p1), dtype)
    nu = jax.random.uniform(k_nu, (b, p2), dtype)
    C = jax.random.uniform(k_c, (b, p1, p2), dtype)
    return mu, nu, C


def _modified_cost(C, u, v, eps):
    # M_ij = (-C_ij + u_i + v_j) / eps, broadcasting like torch unsqueeze(-1)/(-2)
    return (-C + u[..., :, None] + v[..., None, :]) / eps


def workload(mu, nu, C):
    eps = CONFIG['eps']
    log_mu = jnp.log(mu + CONFIG['log_offset'])
    log_nu = jnp.log(nu + CONFIG['log_offset'])

    def body(_, uv):
        u, v = uv
        v = eps * (log_nu - jax.nn.logsumexp(_modified_cost(C, u, v, eps), axis=-2)) + v
        u = eps * (log_mu - jax.nn.logsumexp(_modified_cost(C, u, v, eps), axis=-1)) + u
        return u, v

    u, v = lax.fori_loop(0, CONFIG['max_iter'], body, (jnp.ones_like(mu), jnp.ones_like(nu)))
    pi = jnp.exp(_modified_cost(C, u, v, eps))
    cost = jnp.sum(pi * C, axis=(-2, -1))
    return cost, pi
