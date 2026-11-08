import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


CONFIGS = {
    'dsv4_pro_decode': {
        'name': 'dsv4_pro_decode',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'multi_head_concatenation',
        'num_tokens': 256,
        'hidden_size': 7168,
        'hc_mult': 4,
        'token_block_size': 32,
    },
    'dsv4_pro_prefill': {
        'name': 'dsv4_pro_prefill',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'multi_head_concatenation',
        'num_tokens': 2048,
        'hidden_size': 7168,
        'hc_mult': 4,
        'token_block_size': 32,
    },
    'dsv4_flash_decode': {
        'name': 'dsv4_flash_decode',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'multi_head_concatenation',
        'num_tokens': 256,
        'hidden_size': 4096,
        'hc_mult': 4,
        'token_block_size': 64,
    },
    'dsv4_flash_prefill': {
        'name': 'dsv4_flash_prefill',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'multi_head_concatenation',
        'num_tokens': 2048,
        'hidden_size': 4096,
        'hc_mult': 4,
        'token_block_size': 64,
    },
    'mhc_small': {
        'name': 'mhc_small',
        'model': 'DeepSeek-V4',
        'operator': 'multi_head_concatenation',
        'num_tokens': 128,
        'hidden_size': 2048,
        'hc_mult': 4,
        'token_block_size': 32,
    },
}


CONFIG = CONFIGS['dsv4_pro_decode']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base)."""
  cfg = config or CONFIG
  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

  num_tokens = cfg['num_tokens']
  hidden_size = cfg['hidden_size']
  hc_mult = cfg['hc_mult']
  hc_mult3 = hc_mult * (hc_mult + 2)

  x = jax.random.normal(k1, (num_tokens, hidden_size), dtype=dtype)
  residual = jax.random.normal(
      k2, (num_tokens, hc_mult, hidden_size), dtype=dtype
  )
  post_layer_mix = jax.random.uniform(
      k3, (num_tokens, hc_mult, 1), dtype=jnp.float32
  )
  comb_res_mix = jax.random.uniform(
      k4, (num_tokens, hc_mult, hc_mult), dtype=jnp.float32
  )
  fn = (
      jax.random.normal(
          k5, (hc_mult3, hc_mult * hidden_size), dtype=jnp.float32
      )
      * 0.02
  )
  hc_scale = jax.random.uniform(k6, (3,), dtype=jnp.float32)
  hc_base = jax.random.uniform(k7, (hc_mult3,), dtype=jnp.float32)

  return x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base


import jax
import jax.numpy as jnp


def _mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  num_tokens = mixes.shape[0]

  mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

  pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
  pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

  post_logits = (
      mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
      + hc_base[hc_mult : 2 * hc_mult]
  )
  post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

  comb_logits = mixes[:, 2 * hc_mult :].reshape(
      num_tokens, hc_mult, hc_mult
  ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
  comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
  comb_mix = comb_mix / (
      jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
  )
  for _ in range(sinkhorn_repeat - 1):
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
    )
    comb_mix = comb_mix / (
        jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
    )
  return pre_mix, post_mix, comb_mix


def workload(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base):
  """Reference implementation of fused MHC post/pre in plain JAX."""
  rms_eps = 1e-6
  hc_pre_eps = 1e-6
  hc_sinkhorn_eps = 1e-6
  hc_post_mult_value = 2.0
  sinkhorn_repeat = 20

  outer_shape = residual.shape[:-2]
  hc_mult, hidden_size = residual.shape[-2:]

  x2d = x.reshape(-1, hidden_size).astype(jnp.bfloat16)
  res3d = residual.reshape(-1, hc_mult, hidden_size).astype(jnp.bfloat16)
  post3d = post_layer_mix.reshape(-1, hc_mult, 1).astype(jnp.float32)
  comb3d = comb_res_mix.reshape(-1, hc_mult, hc_mult).astype(jnp.float32)

  # 1. Post-recombine
  x_f32 = x2d[:, None, :].astype(jnp.float32)
  res_f32 = res3d.astype(jnp.float32)

  acc = post3d * x_f32
  for i in range(hc_mult):
    acc = acc + comb3d[:, i, :, None] * res_f32[:, i : i + 1, :]

  new_res3d = acc.astype(jnp.bfloat16)
  new_res2d = new_res3d.reshape(-1, hc_mult * hidden_size)

  # 2. Pre-GEMM and squared sum
  gf = new_res2d.astype(jnp.float32)
  dn = (((1,), (1,)), ((), ()))
  mixes = jax.lax.dot_general(
      gf,
      fn.astype(jnp.float32),
      dn,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )
  sqrsum = jnp.sum(gf * gf, axis=-1, keepdims=True)

  # 3. Gates and Sinkhorn
  pre_mix, post_mix_cur, comb_mix_cur = _mhc_pre_gates(
      mixes,
      sqrsum,
      hc_mult,
      hidden_size,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
  )

  # 4. Collapse streams for next layer input
  gf_3d = gf.reshape(-1, hc_mult, hidden_size)
  lay = pre_mix[:, 0:1] * gf_3d[:, 0, :]
  for i in range(1, hc_mult):
    lay = lay + pre_mix[:, i : i + 1] * gf_3d[:, i, :]
  layer_input_cur = lay.astype(jnp.bfloat16)

  return (
      new_res3d.reshape(*outer_shape, hc_mult, hidden_size),
      post_mix_cur.reshape(*outer_shape, hc_mult, 1),
      comb_mix_cur.reshape(*outer_shape, hc_mult, hc_mult),
      layer_input_cur.reshape(*outer_shape, hidden_size),
  )
