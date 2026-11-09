"""[bf16 variant] SelfAttAggregate: max+sum pool over dim1 -> concat -> self-attention (bmm, softmax, bmm) -> projection by (D,1) weight."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SelfAttAggregate_8134',
    'batch': 16,
    'seq': 16,
    'channels': 256,
    'agg_dim': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    hiddens = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['seq'], CONFIG['channels'], CONFIG['agg_dim']), dtype)
    weight = jax.random.normal(k_w, (CONFIG['agg_dim'], 1), dtype) * 0.02
    return hiddens, weight


def workload(hiddens, weight):
    hp = lax.Precision.HIGHEST
    maxpool = jnp.max(hiddens, axis=1)
    avgpool = jnp.sum(hiddens, axis=1)  # avgsum='sum' default
    agg = jnp.concatenate([avgpool, maxpool], axis=1)  # (B, 2C, D)
    energy = jnp.einsum('bcd,bce->bde', agg, agg, precision=hp)  # (B, D, D)
    attention = jax.nn.softmax(energy, axis=-1)
    weighted = jnp.einsum('bde,bce->bdc', attention, agg, precision=hp)  # (B, D, 2C)
    out = jnp.einsum('bdc,dk->bck', weighted, weight, precision=hp)  # (B, 2C, 1)
    return jnp.squeeze(out, axis=-1)
