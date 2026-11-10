"""TranLayer: swapaxes -> linear (nodes to 1) -> ReLU -> reshape -> linear (embed to 1) -> ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TranLayer_5985',
    'batch': 8,
    'seq': 128,
    'num_nodes': 256,
    'embed_dim': 256,
    'out_features': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_wn, k_bn, k_we, k_be = jax.random.split(key, 5)
    scale = CONFIG['weight_scale']
    out = CONFIG['out_features']
    embed_feat = jax.random.normal(
        k_x,
        (CONFIG['batch'], CONFIG['seq'], CONFIG['num_nodes'], CONFIG['embed_dim']),
        dtype,
    )
    w_nodes = jax.random.normal(k_wn, (CONFIG['num_nodes'], out), dtype) * scale
    b_nodes = jax.random.normal(k_bn, (out,), dtype) * scale
    w_embed = jax.random.normal(k_we, (CONFIG['embed_dim'], out), dtype) * scale
    b_embed = jax.random.normal(k_be, (out,), dtype) * scale
    return embed_feat, w_nodes, b_nodes, w_embed, b_embed


def workload(embed_feat, w_nodes, b_nodes, w_embed, b_embed):
    embed_dim = w_embed.shape[0]
    reshaped = jnp.swapaxes(embed_feat, -2, -1)
    agg_nodes = jax.nn.relu(reshaped @ w_nodes + b_nodes)
    agg_nodes = agg_nodes.reshape(-1, embed_dim)
    agg_embed = jax.nn.relu(agg_nodes @ w_embed + b_embed)
    return agg_embed
