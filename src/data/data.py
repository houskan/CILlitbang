import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

import skimage.io as io
import skimage.transform as trans

import random
import os
import shutil


tmp_train_dir = '../tmp/train/'
tmp_validation_dir = '../tmp/validation/'

train_pairs = []
validation_pairs = []

def get_path_pairs(train_path, image_folder, mask_folder):
    images = []
    masks = []

    images_path = os.path.join(train_path, image_folder)
    mask_path = os.path.join(train_path, mask_folder)
    image_files = os.listdir(images_path)
    image_files.sort()
    mask_files = os.listdir(mask_path)
    mask_files.sort()
    for file in image_files:
        images.append(os.path.join(images_path, file))
    for file in mask_files:
        masks.append(os.path.join(mask_path, file))

    result = []
    for pair in zip(images, masks):
        result.append(pair)

    return result

def delete_old_files(*dirs):
    for dir in dirs:
        for file in dir:
            if os.path.isfile(file): os.remove(file)

def copy_path_pairs(pairs, path):

    # removing all previous files in path
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)

    tmp_img_dir = os.path.join(path, 'images')
    tmp_mask_dir = os.path.join(path, 'groundtruth')

    os.mkdir(tmp_img_dir)
    os.mkdir(tmp_mask_dir)

    delete_old_files(tmp_img_dir, tmp_mask_dir)

    for img_path, mask_path in pairs:
        shutil.copy2(img_path, tmp_img_dir)
        shutil.copy2(mask_path, tmp_mask_dir)


def adjustData(img, mask):
    if np.max(img) > 1.0:
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return (img, mask)


def getTrainGenerators(aug_dict, train_path, image_folder, mask_folder, target_size, batch_size, validation_split, seed):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)

    global train_pairs, validation_pairs
    train_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split), len(path_pairs))]
    validation_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split))]

    copy_path_pairs(train_pairs, tmp_train_dir)
    copy_path_pairs(validation_pairs, tmp_validation_dir)

    train_img_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        tmp_train_dir,
        classes=['images'],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    train_mask_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        tmp_train_dir,
        classes=['groundtruth'],
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    validation_img_generator = ImageDataGenerator().flow_from_directory(
        tmp_validation_dir,
        classes=['images'],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    validation_mask_generator = ImageDataGenerator().flow_from_directory(
        tmp_validation_dir,
        classes=['groundtruth'],
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
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
        # For some reason cv2 does not work here and we have to use skimage io here
        #img = cv2.imread(os.path.join(folder, file))
        #img = img.astype(np.float32)
        #img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

        # Loading image as rgb from file, normalizing it to range [0, 1],
        # (interpolate) resizing it to target size and reshaping it
        img = io.imread(os.path.join(folder, file), as_gray=False)
        img = img / 255.0
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield img
