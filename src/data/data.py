import numpy as np
import random
import cv2
import itertools
import os
import tensorflow as tf
from data.mask_to_submission import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

useAugmentation = True
validation_pairs = []

def get_path_pairs(train_path, image_folder, mask_folder):
    images = []
    masks = []

    images_path = os.path.join(train_path, image_folder)
    mask_path = os.path.join(train_path, mask_folder)
    for file in os.listdir(images_path):
        images.append(os.path.join(images_path, file))
    for file in os.listdir(mask_path):
        masks.append(os.path.join(mask_path, file))

    result = []
    for pair in zip(images, masks):
        result.append(pair)

    return result

def adjustResnetImg(img, input_width, input_height):
    img = cv2.resize(img, (input_width, input_height))
    img = img.astype(np.float32)
    #Subtracting mean from color channels. Values are from keras-segmentation, can maybe be improved (or might not be necessary at all - not tested yet)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]

    return img

def adjustResnetMask(mask, output_width, output_height):
    mask = cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    mask = mask[:, :, 0]
    mask = mask / np.max(mask)
    mask[mask > 0.5] = 1.0
    mask[mask <= 0.5] = 0.0

    return mask



def image_augmentation(img, mask, batch_size, datagen_img, datagen_mask):
    # fits the model on batches with real-time data augmentation:
    img = datagen_img.flow(img, batch_size=batch_size, seed=1).next()
    mask = datagen_mask.flow(mask, batch_size=batch_size, seed=1).next()
    return (img, mask)


def getResnetGenerators(train_path, image_folder, mask_folder, 
                        input_height, input_width, output_height, output_width, n_classes, batch_size, validation_split):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)

    global training_pairs, validation_pairs
    training_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split), len(path_pairs))]
    validation_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split))]

    return (trainResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size), 
        validationResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size))

def trainResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size):

    global training_pairs
    cycle = itertools.cycle(training_pairs)

    aug_dict = dict(rotation_range=30.0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    brightness_range=[0.9, 1.1],
                    fill_mode='reflect')

    datagen_img = ImageDataGenerator(**aug_dict)
    datagen_mask = ImageDataGenerator(**aug_dict)

    while(True):
        X = []
        Y = []
        for _ in range(batch_size):
            img_path, mask_path = next(cycle)

            img = cv2.imread(img_path, 1)
            mask = cv2.imread(mask_path, 1)

            if useAugmentation:
                img = np.reshape(img, (1,) + img.shape)
                mask = np.reshape(mask, (1,) + mask.shape)

                # Using data generators to augment image and mask
                # TODO: (Sebastian) Clean this up!!!
                img, mask = image_augmentation(img, mask, batch_size=1, datagen_img=datagen_img, datagen_mask=datagen_mask)

                # For debugging
                #cv2.imwrite('../data/test_img1.png', img[0])
                #cv2.imwrite('../data/test_mask1.png', mask[0])

                img = img[0]
                mask = mask[0]

            img = adjustResnetImg(img, input_width, input_height)
            mask = adjustResnetMask(mask, output_width, output_height)

            seg_labels = np.zeros((output_height, output_width, n_classes))
            for c in range(n_classes):
                seg_labels[:, :, c] = (mask == c).astype(int)
            seg_labels = np.reshape(seg_labels, (output_width*output_height, n_classes))

            X.append(img)
            Y.append(seg_labels)

        yield np.array(X), np.array(Y)

def validationResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size):

    global validation_pairs
    cycle = itertools.cycle(validation_pairs)

    while(True):
        X = []
        Y = []
        for _ in range(batch_size):
            img_path, mask_path = next(cycle)

            img = cv2.imread(img_path, 1)
            mask = cv2.imread(mask_path, 1)

            img = adjustResnetImg(img, input_width, input_height)
            mask = adjustResnetMask(mask, output_width, output_height)

            seg_labels = np.zeros((output_height, output_width, n_classes))
            for c in range(n_classes):
                seg_labels[:, :, c] = (mask == c).astype(int)
            seg_labels = np.reshape(seg_labels, (output_width*output_height, n_classes))

            X.append(img)
            Y.append(seg_labels)

        yield np.array(X), np.array(Y)

def testResnetGenerator(test_path, image_folder, input_height, input_width):

    folder = os.path.join(test_path, image_folder)
    for file in os.listdir(folder):
        X = []

        img = cv2.imread(os.path.join(folder, file))
        img = adjustResnetImg(img, input_width, input_height)

        X.append(img)

        yield np.array(X)

def saveResnetResult(test_path, images, results, output_height, output_width, n_classes):
    colors = [(0,0,0), (255, 255, 255)]
    resultNames = list(map(lambda x: os.path.join(test_path, 'results', x), images))

    for i, item in enumerate(results):
        result = item.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_arr_c = result[:, :] == c
            seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

        img = cv2.resize(seg_img, (608, 608))
        cv2.imwrite(resultNames[i], img)


def getPatchGenerators(train_path, image_folder, mask_folder,
                        input_height, input_width, output_height, output_width, n_classes, batch_size, validation_split, patch_size=(32,32)):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)

    global training_pairs, validation_pairs
    training_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split), len(path_pairs))]
    validation_pairs = [path_pairs[i] for i in range(int(len(path_pairs)*validation_split))]

    return (trainPatchGenerator(input_height=input_height, input_width=input_width, output_height=output_height, output_width=output_width,
                                patch_size=patch_size, n_classes=n_classes, batch_size=batch_size),
            trainPatchGenerator(input_height=input_height, input_width=input_width, output_height=output_height,
                                output_width=output_width,
                                patch_size=patch_size, n_classes=n_classes, batch_size=batch_size))


def trainPatchGenerator(*, input_height, input_width, output_height, output_width,
                        patch_size = (32,32), n_classes=2, batch_size=1):
    global training_pairs
    cycle = itertools.cycle(training_pairs)

    patch_x, patch_y = patch_size
    while (True):
        X = []
        Y = []

        img_path, mask_path = next(cycle)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        img = adjustResnetImg(img, input_width, input_height)
        mask = adjustResnetMask(mask, output_width, output_height)

        seg_labels = np.zeros((output_height, output_width, n_classes))
        for c in range(n_classes):
            seg_labels[:, :, c] = (mask == c).astype(int)

        patches_X = img.reshape(img.shape[0] // patch_x, patch_x, img.shape[1] // patch_y, patch_y, 3).swapaxes(1, 2)
        patches_Y = seg_labels.reshape(seg_labels.shape[0] // (patch_x // 2), patch_x // 2, seg_labels.shape[1] // (patch_y // 2), patch_y // 2, n_classes).swapaxes(1, 2)

        I = patches_X.shape[0]
        J = patches_Y.shape[1]

        for _ in range(batch_size):
            i = random.randint(0, I-1)
            j = random.randint(0, J-1)


            while(np.mean(patches_Y[i][j]) < 0.1 and random.random() < 0.6):
                i = random.randint(0, I - 1)
                j = random.randint(0, J - 1)

            X.append(patches_X[i][j])
            y = np.reshape(patches_Y[i][j], ((patch_x // 2) * (patch_y // 2), n_classes))
            Y.append(y)
        yield np.array(X), np.array(Y)


def validationPatchGenerator(*, input_height, input_width, output_height, output_width,
                        patch_size=(32, 32), n_classes=2, batch_size=1):
    global validation_pairs
    cycle = itertools.cycle(validation_pairs)

    patch_x, patch_y = patch_size
    while (True):
        X = []
        Y = []

        img_path, mask_path = next(cycle)
        img = cv2.imread(img_path, 1)
        mask = cv2.imread(mask_path, 1)

        img = adjustResnetImg(img, input_width, input_height)
        mask = adjustResnetMask(mask, output_width, output_height)

        seg_labels = np.zeros((output_height, output_width, n_classes))
        for c in range(n_classes):
            seg_labels[:, :, c] = (mask == c).astype(int)

        patches_X = img.reshape(img.shape[0] // patch_x, patch_x, img.shape[1] // patch_y, patch_y, 3).swapaxes(1, 2)
        patches_Y = seg_labels.reshape(seg_labels.shape[0] // (patch_x // 2), patch_x // 2, seg_labels.shape[1] // (patch_y // 2), patch_y // 2, n_classes).swapaxes(1, 2)

        I = patches_X.shape[0]
        J = patches_Y.shape[1]

        for _ in range(batch_size):
            i = random.randint(0, I-1)
            j = random.randint(0, J-1)

            X.append(patches_X[i][j])
            y = np.reshape(patches_Y[i][j], ((patch_x // 2) * (patch_y // 2), n_classes))
            Y.append(y)
        yield np.array(X), np.array(Y)


def testPatchGenerator(test_path, image_folder, input_height, input_width, patch_size=(32, 32)):

    folder = os.path.join(test_path, image_folder)
    patch_x, patch_y = patch_size
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        img = adjustResnetImg(img, input_width, input_height)

        patches = img.reshape(img.shape[0] // patch_x, patch_x, img.shape[1] // patch_y, patch_y, 3).swapaxes(1, 2)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X = []
                patch = patches[i][j]
                X.append(patch)
                yield np.array(X)
