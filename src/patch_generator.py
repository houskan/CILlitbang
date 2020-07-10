import numpy as np
import random
import cv2
import itertools
import os
from data.data import *

train_path = '../data/training/'
test_path = '../data/test/'

DEFAULT_STEP_SIZE = 32

#paramters for train generator
PROBABILITY_TO_IGNORE_NON_ROAD_PATCH = 0.5

def add_image_padding(image, padding):
    if len(image.shape) < 3:
        return np.lib.pad(image, ((padding, padding), (padding, padding)), 'reflect')
    elif len(image.shape) == 3:
        return np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    else:
        assert False, "Expected an image for addImagePadding"


def extract_patch_and_context_for_training(image, groundtruth, step_size,
                                           groundtruth_patch_size=32,
                                           context_patch_size=128):
    """
    :param image: Training image
    :param groundtruth: Corresponding groundtruth
    :param step_size: spacing between patch center (if it is smaller than ground_truth_patch_size,
                        overlapping patches will be extracted -> might be good for training)
    :return: list of patch pairs with shape [(mask, context_patch)]
    """
    padding = (context_patch_size - groundtruth_patch_size) // 2
    w = image.shape[0]
    h = image.shape[1]

    image = add_image_padding(image, padding)

    patches_list = []

    for i in range(padding, h + padding, step_size):
        for j in range(padding, w + padding, step_size):
            i_ = i - padding
            j_ = j - padding
            groundtruth_patch = groundtruth[i_:i_ + groundtruth_patch_size, j_:j_ + groundtruth_patch_size]

            if groundtruth_patch.shape[0] < groundtruth_patch_size or groundtruth_patch.shape[1] < groundtruth_patch_size:
                continue

            context_shift = (context_patch_size - groundtruth_patch_size) // 2
            context_patch = image[i - context_shift:i + groundtruth_patch_size + context_shift, j - context_shift:j + groundtruth_patch_size + context_shift, :]
            patches_list.append((groundtruth_patch, context_patch))

    return patches_list


def extract_patch_and_context_for_testing(image,
                                          groundtruth_patch_size=32,
                                          context_patch_size=128):
    """
    :param image: Testing image
    :param groundtruth_patch_size:
    :param context_patch_size:
    :return: list of context_patches
    """
    padding = (context_patch_size - groundtruth_patch_size) // 2
    w = image.shape[0]
    h = image.shape[1]

    image = add_image_padding(image, padding)

    patches_list = []

    for i in range(padding, h + padding, groundtruth_patch_size):
        for j in range(padding, w + padding, groundtruth_patch_size):
            context_shift = (context_patch_size - groundtruth_patch_size) // 2
            context_patch = image[i - context_shift:i + groundtruth_patch_size + context_shift, j - context_shift:j + groundtruth_patch_size + context_shift, :]
            patches_list.append(context_patch)

    return patches_list


def reconstruct_image_from_patches(patches, patch_size=32, image_size=608):
    """
    :param patches: list of patches in shape (num_patches, patch_size, patch_size)
    """
    patches = np.array(patches)

    assert image_size % patch_size == 0
    assert len(patches.shape) == 3

    dim = image_size // patch_size

    image = np.reshape(patches, (dim, dim, patch_size, patch_size))
    image = np.swapaxes(image, 1, 2)
    image = np.reshape(image, (image_size, image_size))
    return image


def get_patch_generators(train_path, image_folder, mask_folder, groundtruth_patch_size, context_patch_size, image_size,
                         batch_size, validation_split=.1):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)

    global training_pairs, validation_pairs
    training_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split), len(path_pairs))]
    validation_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split))]

    return (train_generator(groundtruth_patch_size=groundtruth_patch_size, context_patch_size=context_patch_size, image_size=image_size, batch_size=batch_size),
            validation_generator(groundtruth_patch_size=groundtruth_patch_size, context_patch_size=context_patch_size, image_size=image_size, batch_size=batch_size))


def train_generator(groundtruth_patch_size, context_patch_size, image_size, batch_size=1):
    """
        Yield: returns #batch_size random patches selected from the a single training image. Then proceeds to next image.
    """

    global training_pairs
    cycle = itertools.cycle(training_pairs)

    while (True):
        X = []
        Y = []

        img_path, mask_path = next(cycle)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        img = cv2.resize(img, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :, 0]
        mask = np.reshape(mask, (image_size, image_size, 1))

        img, mask = adjustData(img, mask)

        step_size = min(groundtruth_patch_size, DEFAULT_STEP_SIZE)
        patches = extract_patch_and_context_for_training(img, mask, step_size=step_size, groundtruth_patch_size=groundtruth_patch_size,
                                                         context_patch_size=context_patch_size)


        """seg_labels = np.zeros((output_height, output_width, n_classes))
        for c in range(n_classes):
            seg_labels[:, :, c] = (mask == c).astype(int)"""


        N = len(patches)
        for _ in range(batch_size):
            i = random.randint(0, N-1)

            mask, context = patches[i]

            while(np.mean(mask) < 0.05 and random.random() < PROBABILITY_TO_IGNORE_NON_ROAD_PATCH):
                i = random.randint(0, N-1)
                mask, context = patches[i]
            X.append(context)
            Y.append(mask)
        yield np.array(X), np.array(Y)


def validation_generator(groundtruth_patch_size, context_patch_size, image_size, batch_size=1):
    """
        Yield: returns #batch_size random patches selected from the a single validation image. Then proceeds to next image.
    """

    global validation_pairs
    cycle = itertools.cycle(validation_pairs)

    while (True):
        X = []
        Y = []

        img_path, mask_path = next(cycle)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        img = cv2.resize(img, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :, 0]
        mask = np.reshape(mask, (image_size, image_size, 1))

        img, mask = adjustData(img, mask)

        step_size = min(groundtruth_patch_size, DEFAULT_STEP_SIZE)
        patches = extract_patch_and_context_for_training(img, mask, step_size=step_size,
                                                         groundtruth_patch_size=groundtruth_patch_size,
                                                         context_patch_size=context_patch_size)
        N = len(patches)

        for _ in range(batch_size):
            i = random.randint(0, N - 1)
            mask, context = patches[i]
            X.append(context)
            Y.append(mask)
        yield np.array(X), np.array(Y)


def test_generator(test_path, image_folder, groundtruth_patch_size, context_patch_size, image_size):
    """
    Yields each image patch by patch
    """

    folder = os.path.join(test_path, image_folder)

    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))

        img = cv2.resize(img, (image_size, image_size))
        img = img / 255.0

        patches = extract_patch_and_context_for_testing(img, groundtruth_patch_size=groundtruth_patch_size, context_patch_size=context_patch_size)
        for context in patches:
            context = np.reshape(context, (1,) + context.shape)
            yield context
