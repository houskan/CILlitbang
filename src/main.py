import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

import datetime

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import data.data
from models.unet import *
from data.data import *
from data.tensorboard_image import *
from data.post_processing import *

predict_best = True
train_model = True
combined_prediction = True

train_path = '../data/training/'
test_path = '../data/test/'
model_path = '../tmp/model.h5'

STEPS_PER_EPOCH = 100
EPOCHS = 50

# Augmentation parameters for training generator (not validation!)
data_gen_args = dict(rotation_range=45,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

# Initializing training and validation generators
train_gen, val_gen = getTrainGenerators(data_gen_args, train_path=train_path,
                                        image_folder='images', mask_folder='groundtruth',
                                        target_size=(400, 400), batch_size=4, validation_split=0.1, seed=2)

print('Keras Version:', keras.__version__)
print('Tensorflow Version:', tf.__version__)

# Initializing and compiling unet model
model = unet()

if train_model:
    # initializing callbacks for training
    callbacks = []

    tensorflow_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-e{}-s{}'.format(EPOCHS, STEPS_PER_EPOCH)

    # Initializing logs directory for tensorboard
    log_dir = '../logs/fit/' + tensorflow_dir
    os.mkdir(log_dir)

    # Initializing tensorboard callback for plots, graph, etc.
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    callbacks.append(tensorboard_callback)

    # Initializing tensorboard image callback for visualizing validation images each epoch
    tensorboard_image_callback = TensorBoardImage(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
    callbacks.append(tensorboard_image_callback)

    # Initialization model checkpoint to store model with best validation loss
    model_checkpoint_callback = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    callbacks.append(model_checkpoint_callback)

    # Training unet model
    model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=val_gen, validation_steps=10, callbacks=callbacks, verbose=1)

# Checking if model weights for best val_loss should be picked for prediction
if predict_best:
    # Loading best result
    model.load_weights(model_path)

if combined_prediction:
    # Saving result masks of test images
    saveCombinedResult(model=model, test_path=test_path, image_folder='images')
else:
    # Initializing test generator
    test_gen = testGenerator(test_path=test_path, image_folder='images', target_size=(400, 400))
    # Predicting results on test images
    images = os.listdir(os.path.join(test_path, 'images'))
    results = model.predict(test_gen, steps=len(images), verbose=1)
    # Saving result masks of test images
    saveResult(test_path=test_path, images=images, results=results)
