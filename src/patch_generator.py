import numpy as np
import random
import cv2
import itertools
import os
from data.data import *

train_path = '../data/training/'
test_path = '../data/test/'


def addImagePadding(image, padding):
    if len(image.shape) < 3:
        return np.lib.pad(image, ((padding, padding), (padding, padding)), 'reflect')
    elif len(image.shape) == 3:
        return np.lib.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    else:
        assert False, "Expected an image for addImagePadding"


def extractPatchAndContextForTraining(image, groundtruth, step_size,
                                      groundtruth_patch_size=16,
                                      local_patch_size=64,
                                      global_patch_size=256):
    """
    :param image: Training image
    :param groundtruth: Corresponding groundtruth
    :param step_size:
    :return: list with shape [(mask, local_patch, global_patch)]
    """
    padding = (global_patch_size - groundtruth_patch_size) // 2
    w = image.shape[0]
    h = image.shape[1]

    image = addImagePadding(image, padding)

    patches_list = []

    for i in range(padding, h + padding, step_size):
        for j in range(padding, w + padding, step_size):
            i_ = i - padding
            j_ = j - padding
            groundtruth_patch = groundtruth[i_:i_ + groundtruth_patch_size, j_:j_ + groundtruth_patch_size]

            if groundtruth_patch.shape[0] < groundtruth_patch_size or groundtruth_patch.shape[1] < groundtruth_patch_size:
                continue

            local_shift = (local_patch_size - groundtruth_patch_size) // 2
            local_patch = image[i - local_shift:i + groundtruth_patch_size + local_shift, j - local_shift:j + groundtruth_patch_size + local_shift, :]
            global_shift = (global_patch_size - groundtruth_patch_size) // 2
            global_patch = image[i - global_shift:i + groundtruth_patch_size + global_shift, j - global_shift:j + groundtruth_patch_size + global_shift, :]
            patches_list.append((groundtruth_patch, local_patch, global_patch))

    return patches_list


def extractPatchesAndContextForTesting(image,
                                       groundtruth_patch_size=16,
                                       local_patch_size=64,
                                       global_patch_size=256):
    """
    :param image: Training image
    :param groundtruth: Corresponding groundtruth
    :return: list with shape[(local_patch, global_patch)]
    """
    padding = (global_patch_size - groundtruth_patch_size) // 2
    w = image.shape[0]
    h = image.shape[1]

    image = addImagePadding(image, padding)

    patches_list = []

    for i in range(padding, h + padding, groundtruth_patch_size):
        for j in range(padding, w + padding, groundtruth_patch_size):
            local_shift = (local_patch_size - groundtruth_patch_size) // 2
            local_patch = image[i - local_shift:i + groundtruth_patch_size + local_shift, j - local_shift:j + groundtruth_patch_size + local_shift, :]
            global_shift = (global_patch_size - groundtruth_patch_size) // 2
            global_patch = image[i - global_shift:i + groundtruth_patch_size + global_shift, j - global_shift:j + groundtruth_patch_size + global_shift, :]
            patches_list.append((local_patch, global_patch))

    return patches_list


def reconstructImageFromPatches(patches, patch_size=16, image_size=608):
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


def getGenerators(train_path, image_folder, mask_folder, groundtruth_patch_size, local_patch_size, global_patch_size,
                       batch_size, validation_split=.1):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)

    global training_pairs, validation_pairs
    training_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split), len(path_pairs))]
    validation_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split))]

    return (trainGenerator(groundtruth_patch_size=groundtruth_patch_size, local_patch_size=local_patch_size, global_patch_size=global_patch_size, batch_size=batch_size),
            validationGenerator(groundtruth_patch_size=groundtruth_patch_size, local_patch_size=local_patch_size, global_patch_size=global_patch_size, batch_size=batch_size))


def trainGenerator(groundtruth_patch_size, local_patch_size, global_patch_size, batch_size=1):
    """
        Yield: returns #batch_size random patches selected from the a single training image. Then proceeds to next image.
    """

    global training_pairs
    cycle = itertools.cycle(training_pairs)

    while (True):
        X1 = []
        X2 = []
        Y = []

        img_path, mask_path = next(cycle)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        img = adjustResnetImg(img, 400, 400)
        mask = adjustResnetMask(mask, 400, 400)

        patches = extractPatchAndContextForTraining(img, mask, step_size=16, groundtruth_patch_size=groundtruth_patch_size,
                                                    local_patch_size=local_patch_size, global_patch_size=global_patch_size)


        """seg_labels = np.zeros((output_height, output_width, n_classes))
        for c in range(n_classes):
            seg_labels[:, :, c] = (mask == c).astype(int)"""


        N = len(patches)
        for _ in range(batch_size):
            i = random.randint(0, N-1)

            mask, local_p, global_p = patches[i]

            while(np.mean(mask) < 0.05 and random.random() < 0.5):
                i = random.randint(0, N-1)
                mask, local_p, global_p = patches[i]
            X1.append(local_p)
            X2.append(global_p)
            Y.append(mask)
        yield [np.array(X1), np.array(X2)], np.array(Y)


def validationGenerator(groundtruth_patch_size, local_patch_size, global_patch_size, batch_size=1):
    """
        Yield: returns #batch_size random patches selected from the a single validation image. Then proceeds to next image.
    """

    global validation_pairs
    cycle = itertools.cycle(validation_pairs)

    while (True):
        X1 = []
        X2 = []
        Y = []

        img_path, mask_path = next(cycle)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        img = adjustResnetImg(img, 400, 400)
        mask = adjustResnetMask(mask, 400, 400)

        patches = extractPatchAndContextForTraining(img, mask, step_size=16,
                                                    groundtruth_patch_size=groundtruth_patch_size,
                                                    local_patch_size=local_patch_size,
                                                    global_patch_size=global_patch_size)

        N = len(patches)

        for _ in range(batch_size):
            i = random.randint(0, N - 1)
            mask, local_p, global_p = patches[i]
            X1.append(local_p)
            X2.append(global_p)
            Y.append(mask)
        yield [np.array(X1), np.array(X2)], np.array(Y)


def testGenerator(test_path, image_folder, groundtruth_patch_size, local_patch_size, global_patch_size):
    """
    Yields each image patch by patch
    """

    folder = os.path.join(test_path, image_folder)

    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        img = adjustResnetImg(img, 608, 608)

        patches = extractPatchesAndContextForTesting(img, groundtruth_patch_size=groundtruth_patch_size, local_patch_size=local_patch_size, global_patch_size=global_patch_size)
        for local_p, global_p in patches:
            X1 = [local_p]
            X2 = [global_p]
            yield [np.array(X1), np.array(X2)]
