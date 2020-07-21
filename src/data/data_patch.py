import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import skimage.io as io
import skimage.transform as trans

import os

from data.helper import *


def add_image_padding(image, padding):
    if len(image.shape) == 3:
        return np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    elif len(image.shape) == 4:
        return np.lib.pad(image, ((0,0), (padding, padding), (padding, padding), (0, 0)), 'reflect')
    else:
        raise Exception("Expected list of images for add_image_padding")


def adjust_data(img, mask):
    if np.max(img) > 1.0:
        img = img / 255.0
    if np.max(mask) > 1.0:
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return img, mask


def getTrainGeneratorsPatch(aug_dict, train_path, validation_path, image_folder, mask_folder, target_size, batch_size, patch_size, seed):

    train_img_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode='rgb',
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    train_mask_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_path,
        classes=[mask_folder],
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

    global train_gen, validation_gen
    train_gen = zip(train_img_generator, train_mask_generator)
    validation_gen = zip(validation_img_generator, validation_mask_generator)

    return train_generator_patch(patch_size), validation_generator_patch(patch_size)


def train_generator_patch(patch_size):
    global train_gen
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        img, mask = extract_random_patch_and_context(img, mask, patch_size)
        yield img, mask


def validation_generator_patch(patch_size):
    global validation_gen
    for (img, mask) in validation_gen:
        img, mask = adjust_data(img, mask)
        img, mask = extract_random_patch_and_context(img, mask, patch_size)
        yield img, mask


def test_generator_patch(test_path, image_dir, target_size, patch_size):
    for file in os.listdir(os.path.join(test_path, image_dir)):
        # Loading image as rgb from file, normalizing it to range [0, 1],
        # (interpolate) resizing it to target size and reshaping it
        img = io.imread(os.path.join(test_path, image_dir, file), as_gray=False)
        if np.max(img) > 1.0:
            img = img / 255.0
        img = trans.resize(img, target_size)

        patches = extract_patch_and_context_for_testing(img, patch_size)
        for context in patches:
            context = np.reshape(context, (1,) + context.shape)
            yield context


def extract_random_patch_and_context(image, groundtruth, patch_size):
    context_patch_size = 4 * patch_size
    padding = (context_patch_size - patch_size) // 2
    num_images = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]

    image = add_image_padding(image, padding)
    X = []
    Y = []
    for index in range(num_images):
        i = np.random.randint(low=padding, high=w+padding - patch_size)
        j = np.random.randint(low=padding, high=h+padding - patch_size)

        i_ = i - padding
        j_ = j - padding
        groundtruth_patch = groundtruth[index, i_:i_ + patch_size, j_:j_ + patch_size]
        context_shift = (context_patch_size - patch_size) // 2
        context_patch = image[index, i - context_shift:i + patch_size + context_shift,
                        j - context_shift:j + patch_size + context_shift, :]
        X.append(context_patch)
        Y.append(groundtruth_patch)

    return np.array(X), np.array(Y)


def extract_patch_and_context_for_testing(image, patch_size):
    context_patch_size = 4 * patch_size
    padding = (context_patch_size - patch_size) // 2
    w = image.shape[0]
    h = image.shape[1]

    image = add_image_padding(image, padding)

    patches_list = []

    for i in range(padding, h + padding, patch_size):
        for j in range(padding, w + padding, patch_size):
            context_shift = (context_patch_size - patch_size) // 2
            context_patch = image[i - context_shift:i + patch_size + context_shift, j - context_shift:j + patch_size + context_shift, :]
            patches_list.append(context_patch)

    return patches_list


def reshape_patch_results_to_images(results, num_images, patch_size, size_adjusted):
    dim = size_adjusted // patch_size
    results = np.reshape(results, (num_images, dim * dim, patch_size, patch_size))
    results = np.reshape(results, (num_images, dim, dim, patch_size, patch_size))
    results = np.swapaxes(results, 2, 3)
    results = np.reshape(results, (num_images, size_adjusted, size_adjusted))
    return results
