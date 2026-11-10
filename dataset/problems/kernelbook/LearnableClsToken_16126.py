"""[bf16 variant] Learnable CLS token: tile + concat onto features, concat False column onto mask, increment lengths."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LearnableClsToken_16126',
    'batch': 64,
    'seq_len': 1024,
    'd_model': 1024,
    'cls_len': 1,
    'fixed_ones_size': 1,
    'min_length': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_features, k_mask, k_lengths, k_cls = jax.random.split(key, 4)
    batch = CONFIG['batch']
    seq_len = CONFIG['seq_len']
    d_model = CONFIG['d_model']
    features = jax.random.uniform(k_features, (batch, seq_len, d_model), dtype)
    mask = jax.random.uniform(k_mask, (batch, seq_len), dtype)
    lengths = jax.random.randint(
        k_lengths, (batch,), CONFIG['min_length'], seq_len + 1, jnp.int32
    )
    cls_param = jax.random.normal(k_cls, (d_model,), dtype) * CONFIG['weight_scale']
    fixed_ones = jnp.ones((CONFIG['fixed_ones_size'],), dtype)
    return features, mask, lengths, cls_param, fixed_ones


def workload(features, mask, lengths, cls_param, fixed_ones):
    batch = features.shape[0]
    cls_tokens = jnp.tile(cls_param[None, None, :], (batch, 1, 1))
    out_features = jnp.concatenate([cls_tokens, features], axis=1)
    zeros = (jnp.tile(fixed_ones[None, :], (batch, 1)) * 0).astype(bool)
    out_mask = jnp.concatenate([zeros, mask], axis=1)
    out_lengths = lengths + CONFIG['cls_len']
    return out_features, out_mask, out_lengths
