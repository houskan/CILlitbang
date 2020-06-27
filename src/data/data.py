import numpy as np
import random
import cv2
import itertools
import os


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

def getResnetGenerators(train_path, image_folder, mask_folder, 
                        input_height, input_width, output_height, output_width, n_classes, batch_size, validation_split):

    path_pairs = get_path_pairs(train_path, image_folder, mask_folder)
    random.shuffle(path_pairs)

    global training_pairs, validation_pairs
    training_pairs = [path_pairs[i] for i in range(int(100*validation_split), len(path_pairs))]
    validation_pairs = [path_pairs[i] for i in range(int(100*validation_split))]

    return (trainResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size), 
        validationResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size))

def trainResnetGenerator(input_height, input_width, output_height, output_width, n_classes, batch_size):

    global training_pairs
    cycle = itertools.cycle(training_pairs)

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
