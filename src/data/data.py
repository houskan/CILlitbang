import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import skimage.io as io
import skimage.transform as trans

import os

from data.helper import *


def adjust_data(img, mask):
    '''Method adjusting images and discrete masks
    :param img: RGB image
    :param mask: segmentation mask
    :return: image/mask pair where the image is in range [0,1] and the mask's pixels are either 0 or
    1
    '''
    if np.max(img) > 1.0:
        img = img / 255.0
    if np.max(mask) > 1.0:
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return img, mask


def train_validation_generators(aug_dict, train_path, validation_path, image_dir, mask_dir, target_size, batch_size, seed):
    '''Method returning generators for the training data and the validation data. Each generator
    returns image/groundtruth pairs
    :param aug_dict: Dictionary describing which augmentations you want to perform. See
    Documentation of ImageDataGenerator of tf.keras
    :param train_path: Path where the training data resides
    :param validation_path: Path where the validation data resides
    :param image_dir: Path relative to train/validation_path. Describes where the RGB images are
    :param mask_dir: Path relative to train/validation_path. Describes where the grayscale images
    are
    :param target_size: What size should the images have
    :param batch_size: How many images per batch
    :param seed: What seed should be used for the shuffle. This is for reproducability
    :return: A train and a validation generator that can be served to TF fit method
    '''

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
    '''Defines the train_generator
    '''
    global train_gen
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield img, mask


def validation_generator():
    '''Defines the validation_generator
    '''
    global validation_gen
    for (img, mask) in validation_gen:
        img, mask = adjust_data(img, mask)
        yield img, mask
