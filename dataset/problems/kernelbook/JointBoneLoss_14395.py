"""JointBoneLoss: pairwise joint gather -> L2 norm over last dim -> |J - Y| -> sum / batch / num_pairs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'JointBoneLoss_14395',
    'batch': 16,
    'joint_num': 32,
    'dim1': 256,
    'dim2': 128,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['joint_num'], CONFIG['dim1'], CONFIG['dim2'])
    joint_out = jax.random.uniform(k1, shape, dtype)
    joint_gt = jax.random.uniform(k2, shape, dtype)
    return joint_out, joint_gt


def workload(joint_out, joint_gt):
    batch, joint_num = joint_out.shape[0], joint_out.shape[1]
    # triu_indices(k=1) yields (i, j) pairs in the same row-major order as the torch double loop.
    id_i, id_j = jnp.triu_indices(joint_num, k=1)
    num_pairs = id_i.shape[0]
    d_out = joint_out[:, id_i] - joint_out[:, id_j]
    d_gt = joint_gt[:, id_i] - joint_gt[:, id_j]
    j_norm = jnp.sqrt(jnp.sum(d_out * d_out, axis=-1))
    y_norm = jnp.sqrt(jnp.sum(d_gt * d_gt, axis=-1))
    return jnp.sum(jnp.abs(j_norm - y_norm)) / batch / num_pairs
