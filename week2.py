import tensorflow as tf
from tensorflow import keras

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

import numpy as np
import matplotlib.pyplot as plt

index = 0

np.set_printoptions(linewidth=320)

print(f'Label: {train_labels[index]}')
print(f'\nImage Pixel Array:\n {train_images[index]}')

train_images = train_images / 255.0
test_images = test_images / 255.0


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.2):
            print('\nLoss is low so cancelling training')
            self.model.stop_training= True

callbacks = myCallback()

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3000, callbacks=callbacks)

print(model.evaluate(test_images, test_labels))
