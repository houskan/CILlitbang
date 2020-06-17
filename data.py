import numpy as np

import os
import glob
import skimage.io as io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import img_as_ubyte

train_path = 'training/training'
test_path = 'test_images'
result_path = 'test_images/results'

def trainGenerator(batch_size = 4):
	image_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['images'], class_mode = None, color_mode = 'rgb', target_size = (400,400), batch_size=batch_size)
	mask_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['groundtruth'], class_mode = None, color_mode = 'grayscale', target_size = (400,400), batch_size=batch_size)
	train_generator = zip(image_generator, mask_generator)
	for (img, mask) in train_generator:
		mask = mask / 255.0
		mask[mask > 0.5] = 1.0
		mask[mask <= 0.5] = 0.0
		img = img / 255.0
		yield(img, mask)

def validationGenerator(batch_size = 2):
	image_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['validation_images'], class_mode = None, color_mode = 'rgb', target_size = (400,400), batch_size=batch_size)
	mask_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['validation_groundtruth'], class_mode = None, color_mode = 'grayscale', target_size = (400,400), batch_size=batch_size)
	validation_generator = zip(image_generator, mask_generator)
	for (img,mask) in validation_generator:
		mask = mask / 255.0
		mask[mask > 0.5] = 1.0
		mask[mask <= 0.5] = 0.0
		img = img / 255.0
		yield(img, mask)

def testGenerator():
	test_generator = ImageDataGenerator().flow_from_directory(test_path, classes = ['test_images'], class_mode = None, color_mode = 'rgb', target_size = (400,400), batch_size = 4)
	for img in test_generator:
		img = img / 255.0
		yield img

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = item[:,:,0] / np.max(item[:,:,0])
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_ubyte(img))
    	

