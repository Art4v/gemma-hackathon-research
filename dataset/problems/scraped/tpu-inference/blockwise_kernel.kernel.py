"""Pallas TPU blockwise quantized matmul (from vllm-project/tpu-inference).

bf16 activations x int8 weights with 128-wide sub-channel weight scales and
on-the-fly int8 activation quantization per block, accumulated in bf16.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = dict(
    n_batch=1024,
    n_in=4096,
    n_out=4096,
    block_size=128,
)

X_Q_DTYPE = jnp.int8
ACC_DTYPE = jnp.bfloat16
MXU_SIZE = 256

BATCH_BLOCK_SIZE = 512
OUT_BLOCK_SIZE = 1024
IN_BLOCK_SIZE = 1024
N_LANE_MULTIPLIER = 1
VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _quantize_block(x, axis, quant_dtype):
    """Symmetric absmax quantization along `axis` (inlined util.quantize_block)."""
    info = jnp.iinfo(quant_dtype)
    dtype_max = float(info.max)
    dtype_min = float(info.min)
    x_max = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    scale = x_max / dtype_max
    x_q = x / scale
    x_q = jnp.round(x_q)
    x_q = jnp.clip(x_q, dtype_min, dtype_max).astype(quant_dtype)
    return x_q, scale


def _unfold_args(conds, values, fn):
    """Turn traced booleans into static Python bools via nested lax.cond."""
    if len(conds) == 0:
        return fn(*values)
    cond, rest = conds[0], conds[1:]
    return jax.lax.cond(
        cond,
        lambda: _unfold_args(rest, values + (True,), fn),
        lambda: _unfold_args(rest, values + (False,), fn),
    )


@functools.partial(jax.jit, static_argnames=["block_size"])
def quantized_matmul_kernel(x, w_q, w_scale, block_size):
    n_batch_total, n_in_total = x.shape
    n_out_total = w_q.shape[0]
    assert n_batch_total % BATCH_BLOCK_SIZE == 0
    assert n_out_total % OUT_BLOCK_SIZE == 0
    assert n_in_total % IN_BLOCK_SIZE == 0
    assert IN_BLOCK_SIZE % block_size == 0

    batch_block_size = BATCH_BLOCK_SIZE
    out_block_size = OUT_BLOCK_SIZE
    in_block_size = IN_BLOCK_SIZE
    x_q_dtype = X_Q_DTYPE
    acc_dtype = ACC_DTYPE

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = n_batch_total // batch_block_size
    n_out = n_out_total // out_block_size
    n_in = n_in_total // in_block_size

    steps_k = in_block_size // block_size
    compute_tile_n = MXU_SIZE * N_LANE_MULTIPLIER
    steps_n = max(1, out_block_size // compute_tile_n)

    def kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch):
        pid_k = pl.program_id(2)
        is_first_step = pid_k == 0
        is_last_step = pid_k == (n_in_total // in_block_size - 1)

        def accum(is_first_step, is_last_step):
            accumulators = [None] * steps_n

            for i in range(steps_k):
                k_start, k_end = i * block_size, (i + 1) * block_size
                lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
                lhs_q, lhs_scale = _quantize_block(lhs_sub, 1, x_q_dtype)
                lhs_scale = lhs_scale.astype(acc_dtype)

                rhs_q_full = rhs_ref[:, k_start:k_end]
                rhs_scale_full = w_scales_ref[i, :, :].astype(acc_dtype)

                for j in range(steps_n):
                    n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n

                    rhs_q_slice = rhs_q_full[n_start:n_end, :]
                    rhs_scale_slice = rhs_scale_full[:, n_start:n_end]
                    dot_res = jax.lax.dot_general(
                        lhs_q,
                        rhs_q_slice,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.int32,
                    )
                    res = dot_res.astype(acc_dtype)
                    res = res * lhs_scale
                    res = res * rhs_scale_slice
                    if i == 0:
                        accumulators[j] = res
                    else:
                        accumulators[j] += res

            acc_block = jnp.concatenate(accumulators, axis=1)

            if not is_first_step:
                acc_block += acc_scratch[...]

            if is_last_step:
                out_ref[...] = acc_block.astype(out_ref.dtype)
            else:
                acc_scratch[...] = acc_block

        _unfold_args((is_first_step, is_last_step), (), accum)

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (batch_block_size, in_block_size),
                    lambda b, o, i: (b, i),
                    memory_space=pltpu.VMEM,
                ),  # x
                pl.BlockSpec(
                    (out_block_size, in_block_size),
                    lambda b, o, i: (o, i),
                    memory_space=pltpu.VMEM,
                ),  # w_q
                pl.BlockSpec(
                    (steps_k, 1, out_block_size),
                    lambda _, o, i: (i, 0, o),
                    memory_space=pltpu.VMEM,
                ),  # w_scale
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                   lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), jnp.bfloat16)
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((n_batch_total, n_out_total), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
    )
    return call(x, w_q, w_scale)


def workload(x, w_q, w_scale):
    """x: [n_batch, n_in]; w_q: [n_out, n_in] int8; w_scale: [nb, 1, n_out] f32."""
    return quantized_matmul_kernel(x, w_q, w_scale, block_size=CONFIG["block_size"])
