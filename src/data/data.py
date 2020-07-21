import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import skimage.io as io
import skimage.transform as trans

import os

from data.helper import *


def adjustData(img, mask):
    if np.max(img) > 1.0:
        img = img / 255.0
    if np.max(mask) > 1.0:
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return img, mask


def getTrainGenerators(aug_dict, train_path, validation_path, image_folder, mask_folder, target_size, batch_size, seed):

    train_img_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_path,
        classes=['images'],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    train_mask_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_path,
        classes=['groundtruth'],
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    validation_img_generator = ImageDataGenerator().flow_from_directory(
        validation_path,
        classes=['images'],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=1,
        seed=seed)
    validation_mask_generator = ImageDataGenerator().flow_from_directory(
        validation_path,
        classes=['groundtruth'],
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=1,
        seed=seed)

    global train_generator, validation_generator
    train_generator = zip(train_img_generator, train_mask_generator)
    validation_generator = zip(validation_img_generator, validation_mask_generator)

    return trainGenerator(), validationGenerator()

def trainGenerator():
    global train_generator
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield img, mask


def validationGenerator():
    global validation_generator
    for (img, mask) in validation_generator:
        img, mask = adjustData(img, mask)
        yield img, mask


def testGenerator(test_path, image_folder, target_size):
    folder = os.path.join(test_path, image_folder)
    for file in os.listdir(folder):
        # Loading image as rgb from file, normalizing it to range [0, 1],
        # (interpolate) resizing it to target size and reshaping it
        img = io.imread(os.path.join(folder, file), as_gray=False)
        if np.max(img) > 1.0:
            img = img / 255.0
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield img
