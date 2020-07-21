import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import skimage.io as io
import skimage.transform as trans

import os

from data.helper import *


def adjust_data(img, mask):
    if np.max(img) > 1.0:
        img = img / 255.0
    if np.max(mask) > 1.0:
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return img, mask


def train_validation_generators(aug_dict, train_path, validation_path, image_dir, mask_dir, target_size, batch_size, seed):

    train_img_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_path,
        classes=[image_dir],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    train_mask_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_path,
        classes=[mask_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    validation_img_generator = ImageDataGenerator().flow_from_directory(
        validation_path,
        classes=[image_dir],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=1,
        seed=seed)
    validation_mask_generator = ImageDataGenerator().flow_from_directory(
        validation_path,
        classes=[mask_dir],
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=1,
        seed=seed)

    global train_gen, validation_gen
    train_gen = zip(train_img_generator, train_mask_generator)
    validation_gen = zip(validation_img_generator, validation_mask_generator)

    return train_generator(), validation_generator()


def train_generator():
    global train_gen
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield img, mask


def validation_generator():
    global validation_gen
    for (img, mask) in validation_gen:
        img, mask = adjust_data(img, mask)
        yield img, mask
