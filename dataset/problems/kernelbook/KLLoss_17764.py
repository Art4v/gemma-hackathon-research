"""[bf16 variant] Symmetric KL loss: cosine-similarity matrices -> /temperature -> softmax -> kl_div(mean) both ways."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'KLLoss_17764',
    'a': 256,
    'b': 128,
    'c': 4,
    'd': 128,
    'temperature': 0.1,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['a'], CONFIG['b'], CONFIG['c'], CONFIG['d'])
    keys = jax.random.split(jax.random.key(0), 4)
    return tuple(jax.random.uniform(k, shape, dtype) for k in keys)


def _softmax_sim(x, y):
    eps = CONFIG['eps']
    xn = x / jnp.maximum(jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True)), eps)
    yn = y / jnp.maximum(jnp.sqrt(jnp.sum(y * y, axis=1, keepdims=True)), eps)
    sim = jnp.einsum('ibcd,jbcd->ijcd', xn, yn, precision=lax.Precision.HIGHEST)
    return jax.nn.softmax(sim / CONFIG['temperature'], axis=1)


def _kl_div_mean(log_input, target):
    # ponytail: target is a softmax with logits in [-1/T, 1/T], so it never hits 0; xlogy not needed
    return jnp.mean(target * (jnp.log(target) - log_input))


def workload(zxs, zys, zxt, zyt):
    sim_s = _softmax_sim(zxs, zys)
    sim_t = _softmax_sim(zxt, zyt)
    loss_s = _kl_div_mean(jnp.log(sim_s), sim_t)
    loss_t = _kl_div_mean(jnp.log(sim_t), sim_s)
    return loss_s, loss_t
