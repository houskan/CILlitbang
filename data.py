import numpy as np
import os
import glob
import skimage.io as io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import img_as_ubyte

train_path = 'training/training'
test_path = 'test_images'
result_path = 'test_images/results'

def trainGenerator():
	image_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['images'], class_mode = None, color_mode = 'rgb', target_size = (400,400), batch_size = 4)
	mask_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['groundtruth'], class_mode = None, color_mode = 'grayscale', target_size = (400,400), batch_size = 4)
	train_generator = zip(image_generator, mask_generator)
	for (img,mask) in train_generator:
		mask = mask / np.max(mask)
		mask[mask > 0.5] = 1
		mask[mask <= 0.5] = 0
		yield(img / 255, mask)

def validationGenerator():
	image_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['validation_images'], class_mode = None, color_mode = 'rgb', target_size = (400,400), batch_size = 2)
	mask_generator = ImageDataGenerator().flow_from_directory(train_path, classes = ['validation_groundtruth'], class_mode = None, color_mode = 'grayscale', target_size = (400,400), batch_size = 2)
	validation_generator = zip(image_generator, mask_generator)
	for (img,mask) in validation_generator:
		mask = mask / np.max(mask)
		mask[mask > 0.5] = 1
		mask[mask <= 0.5] = 0
		yield(img / 255, mask)

def testGenerator():
	test_generator = ImageDataGenerator().flow_from_directory(test_path, classes = ['test_images'], class_mode = None, color_mode = 'rgb', target_size = (400,400), batch_size = 2)
	for img in test_generator:
		yield img / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = item[:,:,0] / np.max(item[:,:,0])
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_ubyte(img))
    	

