"""[bf16 variant] Duel Q-network: Linear+ReLU -> Linear+ReLU -> value/advantage heads -> value + (adv - global mean(adv))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Duel_QNetwork_3783',
    'lead_shape': (8, 128, 128),
    'state_size': 128,
    'hidden': 64,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    s, h, a = CONFIG['state_size'], CONFIG['hidden'], CONFIG['action_size']
    x = jax.random.uniform(keys[0], CONFIG['lead_shape'] + (s,), dtype)
    w1 = jax.random.normal(keys[1], (s, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (h, a), dtype) * 0.02
    bv = jax.random.normal(keys[6], (a,), dtype) * 0.02
    wa = jax.random.normal(keys[7], (h, a), dtype) * 0.02
    ba = jax.random.normal(keys[8], (a,), dtype) * 0.02
    return x, w1, b1, w2, b2, wv, bv, wa, ba


def workload(x, w1, b1, w2, b2, wv, bv, wa, ba):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    value = h @ wv + bv
    adv = h @ wa + ba
    # torch.mean with no dim: one scalar over the whole advantage tensor.
    return value + (adv - jnp.mean(adv))
