import numpy as np

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

epochs = 20
steps_per_epoch = 100
for device in tf.config.experimental.list_physical_devices('GPU)'):
    tf.config.experimental.set_memory_growth(device, True)

print("Keras Version:", keras.__version__)
print("Tensorflow Version:", tf.__version__)

train_path = '../data/training/'
test_path = '../data/test/'

model = resnet50_unet(n_classes=2, input_height=608, input_width=608)

n_classes = model.n_classes
input_height = model.input_height
input_width = model.input_width
output_height = model.output_height
output_width = model.output_width

model.summary()

callbacks = []

# tensorboard initialization
log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
callbacks.append(tensorboard_callback)


trainGen, valGen = getResnetGenerators(train_path=train_path, image_folder='images', mask_folder='groundtruth',
                                       input_height=input_height, input_width=input_width, output_height=output_height,
                                       output_width=output_width,
                                       n_classes=2, batch_size=4, validation_split=0.1)

testGen = testResnetGenerator(test_path=test_path, image_folder='images', input_height=input_height,
                              input_width=input_width)

# tensorboard image initialization
tensorboard_image = TensorBoardImage(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
callbacks.append(tensorboard_image)

model.fit(trainGen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=valGen, validation_steps=10,
          callbacks=callbacks, verbose=1)

images = os.listdir(os.path.join(test_path, 'images'))
results = model.predict(testGen, steps=len(images), verbose=1)
saveResnetResult(test_path=test_path, images=images, results=results, output_height=output_height, output_width=output_width, n_classes=n_classes)

