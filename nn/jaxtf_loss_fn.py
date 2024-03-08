import jax
import tensorflow as tf
from data_creation import x_path, y_path, solver, q0, pi0, time_steps
from jax.experimental import jax2tf

vmaped_solver = jax.vmap(solver.integrate, in_axes=(None, None, None, None, None, 0,))


def solver_fn(jax_embedding_batch):
    return vmaped_solver(
        q0,
        pi0,
        0,
        time_steps,
        'coordinate',
        jax_embedding_batch
    )


converted_integrate = jax2tf.convert(solver_fn)


def slimpletic_loss_fn(y_true, y_pred):
    """
    Loss function for the slimplectic model
    :param y_true: The true embedding
    :param y_pred: The predicted embedding
    :return: The loss
    """
    with tf.device('/CPU:0'):
        # Generate true path
        true_q, true_pi = converted_integrate(y_true)

        # Generate predicted path
        pred_q, pred_pi = converted_integrate(y_pred)

        # Calculate the loss, only in q
        loss = tf.reduce_mean(tf.square(true_q - pred_q))
        return loss


converted_integrate(tf.constant([[1.0, 1.0]]))
