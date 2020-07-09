import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import *
import cv2


from models.model2 import *
from data.data import *
from data.tensorboardimage import *
from patch_generator import *


"""
TODO: This code assume that test_image_size // groundtruth_patch_size == 0. 
If necessary, this can be generalized
"""

epochs = 1
steps_per_epoch = 100
batch_size = 16
validation_split = 0.1


test_image_size = 608
groundtruth_patch_size= 16
local_patch_size = 64
global_patch_size = 256


for device in tf.config.experimental.list_physical_devices('GPU)'):
    tf.config.experimental.set_memory_growth(device, True)

print("Keras Version:", keras.__version__)
print("Tensorflow Version:", tf.__version__)

train_path = '../data/training/'
test_path = '../data/test/'

model = lg_seg_model()
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

#callbacks = []

"""# tensorboard initialization
log_dir = "..\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
callbacks.append(tensorboard_callback)"""


trainGen, valGen = getGenerators(train_path=train_path, image_folder='images', mask_folder='groundtruth',
                                 groundtruth_patch_size=groundtruth_patch_size, local_patch_size=local_patch_size, global_patch_size=global_patch_size,
                                    batch_size=batch_size, validation_split=validation_split)

testGen = testGenerator(test_path=test_path, image_folder='images', groundtruth_patch_size=groundtruth_patch_size, local_patch_size=local_patch_size, global_patch_size=global_patch_size)

"""# tensorboard image initialization
tensorboard_image = TensorBoardImage(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
callbacks.append(tensorboard_image)"""

model.fit(trainGen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=valGen, validation_steps=10, verbose=1)


images = os.listdir(os.path.join(test_path, 'images'))
resultNames = list(map(lambda x: os.path.join(test_path, 'results', x), images))

dim = test_image_size // groundtruth_patch_size

results = model.predict(testGen, steps=len(images)*dim*dim, verbose=1)
results = np.reshape(results, (len(images), dim * dim, groundtruth_patch_size, groundtruth_patch_size))
results = np.reshape(results, (len(images), dim, dim, groundtruth_patch_size, groundtruth_patch_size))
results = np.swapaxes(results, 2, 3)
results = np.reshape(results, (len(images), test_image_size, test_image_size))

for i, item in enumerate(results):
    img = cv2.resize(item, (test_image_size, test_image_size))
    cv2.imwrite(resultNames[i], img)
