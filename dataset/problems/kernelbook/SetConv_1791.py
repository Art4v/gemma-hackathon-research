"""SetConv: four masked-mean-pooled two-layer ReLU MLP set encoders, concat, ReLU MLP, sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SetConv_1791',
    'batch': 64,
    'set_size': 512,
    'sample_feats': 256,
    'predicate_feats': 256,
    'predicate_uri_feats': 256,
    'join_feats': 256,
    'hid_units': 256,
    'out_units': 1,
    'num_sets': 4,
    'weight_scale': 0.02,
}


def _layer_shapes():
    hid = CONFIG['hid_units']
    shapes = []
    for name in ('sample_feats', 'predicate_feats', 'predicate_uri_feats', 'join_feats'):
        shapes.append((CONFIG[name], hid))
        shapes.append((hid, hid))
    shapes.append((CONFIG['num_sets'] * hid, hid))
    shapes.append((hid, CONFIG['out_units']))
    return shapes


def create_inputs(dtype=jnp.float32):
    batch, set_size, hid = CONFIG['batch'], CONFIG['set_size'], CONFIG['hid_units']
    scale = CONFIG['weight_scale']
    feats = (
        CONFIG['sample_feats'],
        CONFIG['predicate_feats'],
        CONFIG['predicate_uri_feats'],
        CONFIG['join_feats'],
    )
    layer_shapes = _layer_shapes()
    num_sets = CONFIG['num_sets']
    keys = jax.random.split(jax.random.key(0), 2 * num_sets + 2 * len(layer_shapes))

    activations = tuple(
        jax.random.uniform(keys[i], (batch, set_size, feats[i]), dtype)
        for i in range(num_sets)
    )
    masks = tuple(
        jax.random.uniform(keys[num_sets + i], (batch, set_size, hid), dtype)
        for i in range(num_sets)
    )

    params = []
    offset = 2 * num_sets
    for i, (n_in, n_out) in enumerate(layer_shapes):
        params.append(jax.random.normal(keys[offset + 2 * i], (n_in, n_out), dtype) * scale)
        params.append(jax.random.normal(keys[offset + 2 * i + 1], (n_out,), dtype) * scale)

    return activations + masks + tuple(params)


def workload(*inputs):
    num_sets = CONFIG['num_sets']
    sets = inputs[:num_sets]
    masks = inputs[num_sets:2 * num_sets]
    params = inputs[2 * num_sets:]

    pooled = []
    for i in range(num_sets):
        w1, b1, w2, b2 = params[4 * i:4 * i + 4]
        hid = jax.nn.relu(sets[i] @ w1 + b1)
        hid = jax.nn.relu(hid @ w2 + b2)
        hid = jnp.sum(hid * masks[i], axis=1) / jnp.sum(masks[i], axis=1)
        pooled.append(hid)

    w1, b1, w2, b2 = params[4 * num_sets:4 * num_sets + 4]
    hid = jnp.concatenate(pooled, axis=1)
    hid = jax.nn.relu(hid @ w1 + b1)
    return jax.nn.sigmoid(hid @ w2 + b2)
