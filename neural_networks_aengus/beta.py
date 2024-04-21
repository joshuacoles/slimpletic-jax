import os

# This guide can only be run with the JAX backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import jax.numpy as jnp

from neural_networks_aengus.data.families import dho
from neural_networks_aengus.our_code_here import rms, vmapped_solve, TRAINING_TIMESTEPS, \
    load_data_wrapped, create_layer

family = dho
loadModelName = "ckpt/model_scaled_2.1.2/checkpoint.model"
newModelName = "model_scaled_2.1.3"
LEARNING_RATE = 1e-7
EPOCHS = 10
NEW = False

isExist = os.path.exists('ckpt')
if not isExist:
    os.makedirs('ckpt', )


@keras.saving.register_keras_serializable()
class PhysicsLoss(keras.layers.Layer):
    """
    A massive hack of adding
    """

    def loss_fn(self, x, y_pred):
        q_true, pi_true = x[:, :, 0], x[:, :, 1]
        q_predicted, pi_predicted = vmapped_solve(y_pred)
        q_predicted = q_predicted.reshape(q_true.shape)
        pi_predicted = pi_predicted.reshape(pi_true.shape)

        rms_q = rms(q_predicted, q_true)
        rms_pi = rms(pi_predicted, pi_true)

        physical_loss = jnp.clip(rms_q, 0, 1e15) + jnp.clip(rms_pi, 0, 1e15)
        # physical_loss = rms_pi + rms_q
        # physical_loss = jnp.clip(rms_pi,0,1e15)
        non_negatives = jnp.mean(jax.lax.select(y_pred < -0.1, 20 * ((y_pred + 0.1) ** 2), jnp.zeros_like(y_pred)))

        return (physical_loss / 2) + jnp.clip(non_negatives, 0, 100)

    # Defines the computation
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.loss_fn(x, y))

        return inputs[1]


def training_loop(family, dataName, model):
    x, y = load_data_wrapped(family, dataName, TRAINING_TIMESTEPS)
    model.fit(x, y, epochs=EPOCHS, batch_size=512, validation_split=0.1, callbacks=[model_checkpoint_callback], )

    return model


def main(initial_model):
    dataList = ["physical-accurate-small-data-0", ]
    for data in dataList:
        initial_model = training_loop(family, data, initial_model)

    return initial_model


checkpoint_filepath = 'ckpt/' + newModelName + '/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

if __name__ == '__main__':
    if NEW == True:
        inputs = keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2))
        lstm_1 = create_layer(50, True)(inputs)
        lstm_2 = create_layer(40, True)(lstm_1)
        lstm_3 = create_layer(20, True)(lstm_2)
        dropout = keras.layers.Dropout(0.15)(lstm_3)
        flatten = keras.layers.Flatten()(dropout)
        dense = keras.layers.Dense(units=dho.embedding_shape[0])(flatten)
        physics_layer = PhysicsLoss()([inputs, dense])

        model = keras.Model(
            inputs=inputs,
            outputs=physics_layer,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=["mae"]
        )

        final_model = main(model)
    else:
        final_model = main(keras.models.load_model(loadModelName + '.keras', custom_objects={
            'PhysicsLoss': PhysicsLoss
        }))

        final_model.save('models/' + newModelName + ".keras")
