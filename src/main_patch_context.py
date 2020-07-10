import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import *
import cv2


from models.unet import *
from data.data import *
from data.tensorboard_image_resnet import *
from patch_generator import *


"""
TODO: This code assume that test_image_size // groundtruth_patch_size == 0. 
If necessary, this can be generalized
"""

EPOCHS = 50
STEPS_PER_EPOCH = 243
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1

GROUNDTRUTH_PATCH_SIZE= 64
CONTEXT_PATCH_SIZE = 64

TRAIN_IMAGE_SIZE=400
TRAIN_IMAGE_SIZE_ADJUSTED=416

TEST_IMAGE_SIZE = 608

if TEST_IMAGE_SIZE % GROUNDTRUTH_PATCH_SIZE != 0:
    TEST_IMAGE_SIZE_ADJUSTED = ((TEST_IMAGE_SIZE // GROUNDTRUTH_PATCH_SIZE) + 1) * GROUNDTRUTH_PATCH_SIZE
else:
    TEST_IMAGE_SIZE_ADJUSTED=TEST_IMAGE_SIZE

assert TEST_IMAGE_SIZE_ADJUSTED % GROUNDTRUTH_PATCH_SIZE == 0

dim = TEST_IMAGE_SIZE_ADJUSTED // GROUNDTRUTH_PATCH_SIZE


for device in tf.config.experimental.list_physical_devices('GPU)'):
    tf.config.experimental.set_memory_growth(device, True)

print("Keras Version:", keras.__version__)
print("Tensorflow Version:", tf.__version__)

train_path = '../data/training/'
test_path = '../data/test/'

model = unet(input_size=(CONTEXT_PATCH_SIZE, CONTEXT_PATCH_SIZE, 3))

model.summary()

#callbacks = []

"""# tensorboard initialization
log_dir = "..\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
callbacks.append(tensorboard_callback)"""


train_gen, validation_gen = get_patch_generators(train_path=train_path, image_folder='images_augmented', mask_folder='groundtruth_augmented',
                                                             groundtruth_patch_size=GROUNDTRUTH_PATCH_SIZE, context_patch_size=CONTEXT_PATCH_SIZE,
                                                             image_size=TRAIN_IMAGE_SIZE_ADJUSTED, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

test_gen = test_generator(test_path=test_path, image_folder='images', groundtruth_patch_size=GROUNDTRUTH_PATCH_SIZE,
                                context_patch_size=CONTEXT_PATCH_SIZE, image_size=TEST_IMAGE_SIZE_ADJUSTED)

"""# tensorboard image initialization
tensorboard_image = TensorBoardImage(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
callbacks.append(tensorboard_image)"""

model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=validation_gen, validation_steps=26, verbose=1)


images = os.listdir(os.path.join(test_path, 'images'))
resultNames = list(map(lambda x: os.path.join(test_path, 'results', x), images))

images_to_classify = len(images)

results = model.predict(test_gen, steps=images_to_classify * dim * dim, verbose=1)
results = np.reshape(results, (images_to_classify, dim * dim, GROUNDTRUTH_PATCH_SIZE, GROUNDTRUTH_PATCH_SIZE))
results = np.reshape(results, (images_to_classify, dim, dim, GROUNDTRUTH_PATCH_SIZE, GROUNDTRUTH_PATCH_SIZE))
results = np.swapaxes(results, 2, 3)
results = np.reshape(results, (images_to_classify, TEST_IMAGE_SIZE_ADJUSTED, TEST_IMAGE_SIZE_ADJUSTED))

for i, item in enumerate(results):
    img = cv2.resize(item, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    img = (img * 255.0).astype('uint8')
    cv2.imwrite(resultNames[i], img)
