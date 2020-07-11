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
from data.tensorboard_image_resnet import *

EPOCHS = 50
STEPS_PER_EPOCH = 100
PATCH_SIZE = (64, 64)
BATCH_SIZE = 12 #here how many patches per image will be extracted
VALIDATION_SPLIT = 0.1

for device in tf.config.experimental.list_physical_devices('GPU)'):
    tf.config.experimental.set_memory_growth(device, True)

print("Keras Version:", keras.__version__)
print("Tensorflow Version:", tf.__version__)

train_path = '../data/training/'
test_path = '../data/test/'

model = resnet50_unet(n_classes=2, input_height=PATCH_SIZE[0], input_width=PATCH_SIZE[1])

n_classes = model.n_classes
input_height = 448
input_width = 448
output_height = 224
output_width = 224

model.summary()

callbacks = []

# tensorboard initialization
log_dir = "..\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
callbacks.append(tensorboard_callback)


trainGen, valGen = getPatchGenerators(train_path=train_path, image_folder='images', mask_folder='groundtruth',
                                      input_height=input_height, input_width=input_width, output_height=output_height,
                                      output_width=output_width, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,
                                      n_classes=n_classes, validation_split=VALIDATION_SPLIT)

testGen = testPatchGenerator(test_path=test_path, image_folder='images', input_height=640, input_width=640,
                             patch_size=PATCH_SIZE)

# tensorboard image initialization
tensorboard_image = TensorBoardImageResnet(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
callbacks.append(tensorboard_image)

model.fit(trainGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=valGen, validation_steps=10,
          callbacks=callbacks, verbose=1)


images = os.listdir(os.path.join(test_path, 'images'))

#values are hardcoded to work with 32x32 patches
#model.predict returns all patches vectorized. the reshapes reconstruct each image from 19*19 patches
#TODO: generalize

results = model.predict(testGen, steps=len(images)*100, verbose=1)
results = np.reshape(results, (len(images), 100, 32, 32, n_classes))
results = np.reshape(results, (len(images), 10, 10, 32, 32, n_classes))
results = np.swapaxes(results, 2, 3)
results = np.reshape(results, (len(images), 320, 320, n_classes))
results = np.reshape(results, (len(images), 320*320, n_classes))
saveResnetResult(test_path=test_path, images=images, results=results, output_height=320, output_width=320, n_classes=n_classes)


