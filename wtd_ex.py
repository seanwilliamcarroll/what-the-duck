from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

# PATH = os.getcwd()
# train_dir = os.path.join(PATH, 'training')
# validation_dir = os.path.join(PATH, 'validation')

# train_ducks_dir = os.path.join(train_dir, 'duck')
# train_notducks_dir = os.path.join(train_dir, 'notduck')
# validation_ducks_dir = os.path.join(validation_dir, 'duck')
# validation_notducks_dir = os.path.join(validation_dir, 'notduck')


# num_ducks_tr = len(os.listdir(train_ducks_dir))
# num_notducks_tr = len(os.listdir(train_notducks_dir))

# num_ducks_val = len(os.listdir(validation_ducks_dir))
# num_notducks_val = len(os.listdir(validation_notducks_dir))

# total_train = num_ducks_tr + num_notducks_tr
# total_val = num_ducks_val + num_notducks_val


# print('total training cat images:', num_ducks_tr)
# print('total training dog images:', num_notducks_tr)

# print('total validation cat images:', num_ducks_val)
# print('total validation dog images:', num_notducks_val)
# print("--")
# print("Total training images:", total_train)
# print("Total validation images:", total_val)

os.environ["CUDA_VISIBLE_DEVICES"]=""
batch_size = 256
epochs = 200
IMG_HEIGHT = 250
IMG_WIDTH = 250

# train_image_generator = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=45,
#     width_shift_range=.15,
#     height_shift_range=.15,
#     horizontal_flip=True,
#     zoom_range=0.5
#                     )
# validation_trans_args = dict(rescale=1./255)
# validation_image_generator = ImageDataGenerator(**validation_trans_args) # Generator for our validation data

# train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
#                                                            directory=train_dir,
#                                                            shuffle=True,
#                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                            class_mode='categorical')

# val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
#                                                               directory=validation_dir,
#                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                               class_mode='categorical')
# # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

new_model = tf.keras.models.load_model('wtd_model.h5')
# new_model.summary()

def get_labels(img_arr):
    output = np.argmax(new_model.predict(img_arr), axis=1)
    return output
def get_label(output):
    if output == 0:
        return 'duck'
    else:
        return 'notduck'

# sample_training_images, _ = next(val_data_gen)
# print(sample_training_images[:1])
# for img in sample_training_images[:1]:
#     print(img.shape)
# print(sample_training_images[:1].shape)
# print(new_model.predict(sample_training_images[:1]))
# print(sample_training_images[:1])

def plotImage(img_arr, img):
    fig = plt.figure(figsize=(20,20))
    label = get_labels(img_arr)[0]
    # print(img.shape)
    plt.imshow(img)
    plt.axis('off')
    plt.title(get_label(label))
    # plt.tight_layout()
    plt.show()

def plotImages(images_arr):
    sq = int(np.ceil(np.sqrt(len(images_arr))))
    fig, axes = plt.subplots(sq, sq, figsize=(20,20))
    axes = axes.flatten()
    labels = get_labels(images_arr)
    # print(labels)
    for img, ax, label in zip( images_arr, axes, labels):
        ax.imshow(img)
        ax.axis('off')
        ax.title.set_text(get_label(label))
    plt.tight_layout()
    plt.show()
# plotImages(sample_training_images[:25])

# # score = new_model.evaluate_generator(val_data_gen, verbose=2)
# # print(score)



# print(new_model.predict(sample_training_images[:10]))

image_string = 'cat_test.jpg'

test_image = image.load_img(image_string, target_size=[IMG_WIDTH, IMG_HEIGHT])
orig_image = image.load_img(image_string)
trans_args = dict(rescale=1./255)
# image_generator = ImageDataGenerator(**validation_trans_args) # Generator for our validation data

# # df = pd.DataFrame({"directory"})
# # image_generator.flow()

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
# print(test_image.shape)
# plotImages([test_image])
plotImage(test_image, orig_image)
