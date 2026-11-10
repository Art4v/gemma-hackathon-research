"""[bf16 variant] SkillDiscriminator: Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear, returning logits and LogSoftmax over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SkillDiscriminator_10260',
    'batch': 8,
    'dim1': 256,
    'dim2': 256,
    'state_dim': 128,
    'hidden_dim': 256,
    'nb_skills': 128,
    'negative_slope': 0.2,
    'log_softmax_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    state_dim = CONFIG['state_dim']
    hidden_dim = CONFIG['hidden_dim']
    nb_skills = CONFIG['nb_skills']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], state_dim),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (state_dim, hidden_dim), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden_dim,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden_dim, hidden_dim), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden_dim,), dtype) * scale
    w3 = jax.random.normal(keys[5], (hidden_dim, nb_skills), dtype) * scale
    b3 = jax.random.normal(keys[6], (nb_skills,), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    slope = CONFIG['negative_slope']
    h = jax.nn.leaky_relu(x @ w1 + b1, slope)
    h = jax.nn.leaky_relu(h @ w2 + b2, slope)
    logits = h @ w3 + b3
    return logits, jax.nn.log_softmax(logits, axis=CONFIG['log_softmax_axis'])
