import os

# This guide can only be run with the jax backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
from our_code_here import get_data, get_loss_fn, get_model

# Prepare the training dataset.
batch_size = 32

train_dataset, val_dataset = get_data(batch_size)
loss_fn = get_loss_fn()
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


def compute_loss_and_updates(
        trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (non_trainable_variables, metric_variables)


grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)


@jax.jit
def train_step(state, data):
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = state
    x, y = data
    (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
        trainable_variables, non_trainable_variables, metric_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # Return updated state
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )


@jax.jit
def eval_step(state, data):
    trainable_variables, non_trainable_variables, metric_variables = state
    x, y = data
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = val_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (
        trainable_variables,
        non_trainable_variables,
        metric_variables,
    )


# Build optimizer variables.
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
metric_variables = train_acc_metric.variables
state = (
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
)

# Training loop
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # Log every 100 batches.
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, _, metric_variables = state
        for variable, value in zip(train_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Training accuracy: {train_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")

metric_variables = val_acc_metric.variables

(
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
) = state

state = (trainable_variables, non_trainable_variables, metric_variables)

# Eval loop
for step, data in enumerate(val_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = eval_step(state, data)
    # Log every 100 batches.
    if step % 100 == 0:
        print(f"Validation loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, metric_variables = state
        for variable, value in zip(val_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Validation accuracy: {val_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")
