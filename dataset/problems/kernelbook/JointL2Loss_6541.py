"""JointL2Loss: reshape to (B*J, -1) -> sum((gt - pred)^2, axis=1) -> sqrt -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'JointL2Loss_6541',
    'batch': 16,
    'joint_num': 16384,
    'joint_split': 4,
    'dim': 128,
}


def create_inputs(dtype=jnp.float32):
    b, j, s, d = CONFIG['batch'], CONFIG['joint_num'], CONFIG['joint_split'], CONFIG['dim']
    k_pred, k_gt = jax.random.split(jax.random.key(0))
    joint_pred = jax.random.uniform(k_pred, (b, j // s, s, d), dtype)
    joint_gt = jax.random.uniform(k_gt, (b, j, d), dtype)
    return joint_pred, joint_gt


def workload(joint_pred, joint_gt):
    batch_size, joint_num, _ = joint_gt.shape
    pred = joint_pred.reshape(batch_size * joint_num, -1)
    gt = joint_gt.reshape(batch_size * joint_num, -1)
    offset = jnp.sum(jnp.square(gt - pred), axis=1)
    return jnp.mean(jnp.sqrt(offset))
