"""[bf16 variant] BitNet b1.58 BitLinear: ternary absmean weight quant + per-token absmax int8 activation quant + linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'bitnet_ternary_bitlinear',
    'batch': 8,
    'seq_len': 512,
    'in_features': 4096,
    'out_features': 11008,
    'weight_eps': 1e-5,
    'act_eps': 1e-5,
    'act_qmax': 127.0,
    'act_qmin': -128.0,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    return x, w


def _weight_quant(w):
    wf = w.astype(jnp.float32)
    scale = 1.0 / jnp.maximum(jnp.mean(jnp.abs(wf)), CONFIG['weight_eps'])
    return (jnp.clip(jnp.round(wf * scale), -1.0, 1.0) / scale).astype(w.dtype)


def _act_quant(x):
    xf = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
    scale = CONFIG['act_qmax'] / jnp.maximum(absmax, CONFIG['act_eps'])
    q = jnp.clip(jnp.round(xf * scale), CONFIG['act_qmin'], CONFIG['act_qmax'])
    return (q / scale).astype(x.dtype)


def workload(x, w):
    return _act_quant(x) @ _weight_quant(w)
