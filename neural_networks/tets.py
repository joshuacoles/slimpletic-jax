import tensorflow as tf

from neural_networks.data import load_nn_data

x_data, y_data = load_nn_data('dho', 'physical-accurate-0')

original = tf.data.TFRecordDataset.from_tensor_slices(
    (x_data, y_data)
)

original = original
original.save('test.tfrecord')

new = (
    tf.data.TFRecordDataset.load('test.tfrecord')
    .shuffle(buffer_size=1024)
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
    .batch(32)
)

print(new)
