import numpy as np
import random
import cv2
import itertools
import os

from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans


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

def adjustData(img, mask):
    if (np.max(img) > 1.0):
        img = img / 255.0
        mask = mask / np.max(mask)
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return (img, mask)

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

def getTrainGenerators(aug_dict, train_path, test_path, batch_size=4, image_color_mode='rgb',
                       mask_color_mode='grayscale', target_size=(400, 400), seed=1):
    global train_generator, validation_generator

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=['images'],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        subset='training')
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=['groundtruth'],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        subset='training')
    v_image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=['images'],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        subset='validation')
    v_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=['groundtruth'],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed,
        subset='validation')

    train_generator = zip(image_generator, mask_generator)
    validation_generator = zip(v_image_generator, v_mask_generator)
    return trainGenerator(), validationGenerator()


def trainGenerator():
    global train_generator
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)

def trainResnetGenerator(batch_size, train_path, image_folder, mask_folder, n_classes,
        input_height, input_width, output_height, output_width):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)
    cycle = itertools.cycle(path_pairs)

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


def validationGenerator():
    global validation_generator
    for (img, mask) in validation_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def testGenerator(test_path, num_image=10, target_size=(400, 400), image_color_mode='rgb'):
    test_path = os.path.join(test_path, 'images')
    dirs = os.listdir(test_path)
    for i, file in zip(range(len(dirs)), dirs):
        img = io.imread(os.path.join(test_path, file), as_gray=(False if image_color_mode == 'rgb' else True))
        img = img / 255.0
        img = trans.resize(img, target_size)
        # img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img

def testResnetGenerator(test_path, image_folder, input_height, input_width):

    folder = os.path.join(test_path, image_folder)
    for file in os.listdir(folder):
        X = []

        img = cv2.imread(os.path.join(folder, file))
        img = adjustResnetImg(img, input_width, input_height)

        X.append(img)

        yield np.array(X)

def saveResult(test_path, npyfile):
    images = os.listdir(os.path.join(test_path, 'images'))
    results = list(map(lambda x: os.path.join(test_path, 'results', x), images))
    
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(results[i], img)

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