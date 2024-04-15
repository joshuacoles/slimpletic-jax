import datetime
import json
import os
from pathlib import Path

from neural_networks.data import project_data_root

# This guide can only be run with the jax backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

import keras
from our_code_here import get_data, loss_fn, get_model, EPOCHS, BATCH_SIZE, dataName, family, SHUFFLE_SEED, \
    TRAINING_TIMESTEPS

train_dataset, val_dataset = get_data(BATCH_SIZE)
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


def compute_loss_and_updates(
        trainable_variables, non_trainable_variables, metric_variables, x, y
):
    # jax.debug.print("trainable_variables {}", non_trainable_variables)
    # jax.debug.print("non_trainable_variables {}", non_trainable_variables)

    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )

    # jax.debug.print("post stateless_call non_trainable_variables {}", non_trainable_variables)

    loss = loss_fn(
        true_embedding=y,
        true_trajectory=x,
        predicted_embedding=y_pred
    )

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

    loss = loss_fn(
        true_embedding=y,
        true_trajectory=x,
        predicted_embedding=y_pred
    )

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

data_dir = project_data_root.joinpath("training_data").joinpath(datetime.datetime.now().isoformat())
data_dir.mkdir(parents=True)
loss_data = open(data_dir.joinpath("loss.csv"), "w")
val_loss_data = open(data_dir.joinpath("val_loss.csv"), "w")

loss_data.write("epoch,step,loss,accuracy\n")
val_loss_data.write("epoch,step,loss,accuracy\n")

json.dump({
    "data_name": dataName,
    "family": family.key,
    "epochs": EPOCHS,
    "training_timesteps": TRAINING_TIMESTEPS,
    "batch_size": BATCH_SIZE,
    "shuffle_seed": SHUFFLE_SEED,
}, open(data_dir.joinpath("config.json"), "w"))

for epoch in range(EPOCHS):
    # Training loop
    for step, data in enumerate(train_dataset):
        data = (data[0].numpy(), data[1].numpy())
        loss, state = train_step(state, data)

        if step % 1000 == 0:
            print(f"{epoch}: Training loss (for 1 batch) at batch {step}: {float(loss):.4f}")
            _, _, _, metric_variables = state
            for variable, value in zip(train_acc_metric.variables, metric_variables):
                variable.assign(value)
            print(f"{epoch}: Training accuracy: {train_acc_metric.result()}")
            print(f"{epoch}: Seen so far: {(step + 1) * BATCH_SIZE} samples")
            loss_data.write(f"{epoch},{step},{float(loss)},{train_acc_metric.result()}\n")
            loss_data.flush()

    metric_variables = val_acc_metric.variables

    print(metric_variables)

    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = state

    val_state = (trainable_variables, non_trainable_variables, metric_variables)

    # Eval loop
    for step, data in enumerate(val_dataset):
        data = (data[0].numpy(), data[1].numpy())
        loss, val_state = eval_step(val_state, data)
        # Log every 100 batches.
        if step % 100 == 0:
            print(f"Validation loss (for 1 batch) at batch {step}: {float(loss):.4f}")
            _, _, metric_variables = val_state
            for variable, value in zip(val_acc_metric.variables, metric_variables):
                variable.assign(value)
            print(f"Validation accuracy: {val_acc_metric.result()}")
            print(f"Seen so far: {(step + 1) * BATCH_SIZE} samples")
            val_loss_data.write(f"{epoch},{step},{float(loss)},{val_acc_metric.result()}\n")
            val_loss_data.flush()

    # Save model
    keras.models.save_model(model, data_dir.joinpath(f"model_{epoch}.keras"))
