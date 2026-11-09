"""[bf16 variant] Discriminator MLP: concat -> Linear+tanh x3 -> Linear+sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_4203',
    'batch': 65536,
    'state_dim': 512,
    'action_dim': 512,
    'hidden1': 500,
    'hidden2': 300,
    'hidden3': 300,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 6)
    b = CONFIG['batch']
    in_dim = CONFIG['state_dim'] + CONFIG['action_dim']
    dims = [in_dim, CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['hidden3'], CONFIG['out_dim']]
    state = jax.random.uniform(keys[0], (b, CONFIG['state_dim']), dtype)
    action = jax.random.uniform(keys[1], (b, CONFIG['action_dim']), dtype)
    params = []
    for i in range(4):
        w = jax.random.normal(keys[2 + i], (dims[i], dims[i + 1]), dtype) * 0.02
        params += [w, jnp.zeros((dims[i + 1],), dtype)]
    return (state, action, *params)


def workload(state, action, w1, b1, w2, b2, w3, b3, w4, b4):
    x = jnp.concatenate([state, action], axis=1)
    x = jnp.tanh(x @ w1 + b1)
    x = jnp.tanh(x @ w2 + b2)
    x = jnp.tanh(x @ w3 + b3)
    return jax.nn.sigmoid(x @ w4 + b4)
