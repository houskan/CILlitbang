import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import data.data
from models.unet import *
from data.data import *
from data.tensorboard_image import *
from data.post_processing import *

import datetime
import argparser

parser = argparser.get_parser()
args = parser.parse_args()

# Augmentation parameters for training generator (not validation!)
data_gen_args = dict(rotation_range=360,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

# Initializing training and validation generators
train_gen, val_gen = getTrainGenerators(data_gen_args,
                                        train_path=args.train_path, validation_path=args.val_path,
                                        image_folder='images', mask_folder='groundtruth',
                                        target_size=(400, 400), batch_size=4, seed=1)

print('Keras Version:', keras.__version__)
print('Tensorflow Version:', tf.__version__)

# Initializing and compiling unet model
model = unet_dilated2()

if args.train_model:
    # Initializing callbacks for training
    callbacks = []

    tensorflow_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-e{}-s{}'.format(args.epochs, args.steps)

    # Initializing logs directory for tensorboard
    log_dir = os.path.join('../logs/fit', tensorflow_dir)
    os.mkdir(log_dir)

    # Initializing tensorboard callback for plots, graph, etc.
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    callbacks.append(tensorboard_callback)

    # Initializing tensorboard image callback for visualizing validation images each epoch
    tensorboard_image_callback = TensorBoardImage(log_dir=log_dir, validation_pairs=data.data.validation_pairs)
    callbacks.append(tensorboard_image_callback)

    # Initialization model checkpoint to store model with best validation loss
    model_checkpoint_callback = ModelCheckpoint(args.model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    callbacks.append(model_checkpoint_callback)

    # Training unet model
    model.fit(train_gen, steps_per_epoch=args.steps, epochs=args.epochs, validation_data=val_gen, validation_steps=10, callbacks=callbacks, verbose=1)

# Checking if model weights for best val_loss should be picked for prediction
if args.predict_best:
    # Loading best result
    model.load_weights(model_path)

if args.comb_pred:
    # Saving result masks of test images
    saveCombinedResult(model=model, test_path=test_path, image_folder='images')
else:
    # Initializing test generator
    test_gen = testGenerator(test_path=args.test_path, image_folder='images', target_size=(400, 400))
    # Predicting results on test images
    images = os.listdir(os.path.join(test_path, 'images'))
    results = model.predict(test_gen, steps=len(images), verbose=1)
    # Saving result masks of test images
    saveResult(test_path=args.test_path, images=images, results=results)
