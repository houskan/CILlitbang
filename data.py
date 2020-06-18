import numpy as np

import os
import glob
import skimage.io as io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import img_as_ubyte
import skimage.io as io
import skimage.transform as trans


train_path = 'training/training'
test_path = 'test_images'
result_path = 'test_images/results'

def adjustData(img, mask):
    if (np.max(img) > 1.0):
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
    return (img, mask)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode='rgb', mask_color_mode='grayscale', target_size=(400, 400), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)

def testGenerator(test_path, num_image = 10, target_size = (400, 400), image_color_mode ='rgb'):
    dirs = os.listdir(test_path)
    for i, file in zip(range(num_image), dirs):
        img = io.imread(os.path.join(test_path, file), as_gray = (False if image_color_mode == 'rgb' else True))
        img = img / 255.0
        img = trans.resize(img, target_size)
        #img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

def saveResult(save_path, npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path, "%d_predict.png"%i), img)
