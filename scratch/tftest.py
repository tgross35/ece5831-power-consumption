import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y


def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return (
        tf.data.Dataset.from_tensor_slices((xs, ys))
        .map(preprocess)
        .shuffle(len(ys))
        .batch(128)
    )


print("Creating training dataset")
train_dataset = create_dataset(x_train, y_train)

print("Creating validation dataset")
val_dataset = create_dataset(x_val, y_val)

print("Creating model")
model = keras.Sequential(
    [
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(units=256, activation="relu"),
        keras.layers.Dense(units=192, activation="relu"),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dense(units=10, activation="softmax"),
    ]
)

print("Compiling model")
model.compile(
    optimizer="adam",
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("Fitting model")
history = model.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data=val_dataset.repeat(),
    validation_steps=2,
)

print("Forming predictions")
predictions = model.predict(val_dataset)

import pdb

pdb.set_trace()
