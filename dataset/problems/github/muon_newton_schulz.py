"""[bf16 variant] Muon step: nesterov momentum lerp -> Frobenius normalize -> 5 quintic Newton-Schulz iterations -> aspect scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'muon_newton_schulz',
    'batch': 16,
    'rows': 1024,
    'cols': 4096,
    'steps': 5,
    'beta': 0.95,
    'ns_a': 3.4445,
    'ns_b': -4.7750,
    'ns_c': 2.0315,
    'eps': 1e-7,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['rows'], CONFIG['cols'])
    grad = jax.random.normal(k1, shape, dtype)
    momentum = jax.random.normal(k2, shape, dtype)
    return grad, momentum


def _newton_schulz5(g):
    a, b, c = CONFIG['ns_a'], CONFIG['ns_b'], CONFIG['ns_c']
    tall = g.shape[-2] > g.shape[-1]
    x = g.astype(jnp.float32)
    if tall:
        x = jnp.swapaxes(x, -1, -2)
    norm = jnp.sqrt(jnp.sum(x * x, axis=(-2, -1), keepdims=True))
    x = x / (norm + CONFIG['eps'])
    for _ in range(CONFIG['steps']):
        xt = jnp.swapaxes(x, -1, -2)
        A = jnp.matmul(x, xt, precision=_HI)
        B = b * A + jnp.matmul(c * A, A, precision=_HI)
        x = a * x + jnp.matmul(B, x, precision=_HI)
    if tall:
        x = jnp.swapaxes(x, -1, -2)
    return x


def workload(grad, momentum):
    beta = CONFIG['beta']
    momentum = momentum + (1.0 - beta) * (grad - momentum)
    update = grad + beta * (momentum - grad)
    update = _newton_schulz5(update)
    update = update * max(1.0, grad.shape[-2] / grad.shape[-1]) ** 0.5
    return update, momentum
