from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

PATH = os.getcwd()
train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')

train_ducks_dir = os.path.join(train_dir, 'duck')
train_notducks_dir = os.path.join(train_dir, 'notduck')
validation_ducks_dir = os.path.join(validation_dir, 'duck')
validation_notducks_dir = os.path.join(validation_dir, 'notduck')


num_ducks_tr = len(os.listdir(train_ducks_dir))
num_notducks_tr = len(os.listdir(train_notducks_dir))

num_ducks_val = len(os.listdir(validation_ducks_dir))
num_notducks_val = len(os.listdir(validation_notducks_dir))

total_train = num_ducks_tr + num_notducks_tr
total_val = num_ducks_val + num_notducks_val


print('total training cat images:', num_ducks_tr)
print('total training dog images:', num_notducks_tr)

print('total validation cat images:', num_ducks_val)
print('total validation dog images:', num_notducks_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


batch_size = 256
epochs = 200
IMG_HEIGHT = 250
IMG_WIDTH = 250

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
                    )
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# sample_training_images, _ = next(train_data_gen)
# plotImages(sample_training_images[:5])
# Include the epoch in the file name (uses `str.format`)

checkpoint_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.mkdir("checkpoint/" + checkpoint_dir)
checkpoint_path = "checkpoint/" + checkpoint_dir + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
latest = tf.train.latest_checkpoint("checkpoint/2019-12-19_23-20-48/")
print(latest)
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('wtd_model.h5')

# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# score = model.evaluate_generator(val_data_gen)
# print(score)
# model.save('my_model.h5')
