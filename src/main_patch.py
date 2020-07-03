import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from keras.callbacks import TensorBoard
import cv2

import datetime

import data.data
from models.model import *
from data.data import *
from data.tensorboardimage import *

epochs = 50
steps_per_epoch = 100
patch_size = (32, 32)
batch_size = 16 #here how many patches per image will be extracted
validation_split = 0.1

for device in tf.config.experimental.list_physical_devices('GPU)'):
    tf.config.experimental.set_memory_growth(device, True)

print("Keras Version:", keras.__version__)
print("Tensorflow Version:", tf.__version__)

train_path = '../data/training/'
test_path = '../data/test/'

model = resnet50_unet(n_classes=2, input_height=patch_size[0], input_width=patch_size[1])

n_classes = model.n_classes
input_height = 416
input_width = 416
output_height = 208
output_width = 208

model.summary()

callbacks = []

# tensorboard initialization
log_dir = "..\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
callbacks.append(tensorboard_callback)


trainGen, valGen = getPatchGenerators(train_path=train_path, image_folder='images', mask_folder='groundtruth',
                                       input_height=input_height, input_width=input_width, output_height=output_height,
                                       output_width=output_width, batch_size=batch_size, patch_size=patch_size,
                                       n_classes=n_classes,  validation_split=validation_split)

testGen = testPatchGenerator(test_path=test_path, image_folder='images', input_height=608, input_width=608,
                             patch_size=patch_size)

# tensorboard image initialization
tensorboard_image = TensorBoardImage(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
callbacks.append(tensorboard_image)

model.fit(trainGen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=valGen, validation_steps=10,
          callbacks=callbacks, verbose=1)


images = os.listdir(os.path.join(test_path, 'images'))

#values are hardcoded to work with 32x32 patches
#model.predict returns all patches vectorized. the reshapes reconstruct each image from 19*19 patches
#TODO: generalize

results = model.predict(testGen, steps=len(images)*19*19, verbose=1)
results = np.reshape(results, (len(images), 361, 16, 16, n_classes))
results = np.reshape(results, (len(images), 19, 19, 16, 16, n_classes))
results = np.swapaxes(results, 2, 3)
results = np.reshape(results, (len(images), 304, 304, n_classes))
results = np.reshape(results, (len(images), 304*304, n_classes))
saveResnetResult(test_path=test_path, images=images, results=results, output_height=304, output_width=304, n_classes=n_classes)


