import argparser
import datetime
import numpy as np
import numpy.random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from models.unet import *
from models.unet_dilated_v1 import *
from models.unet_dilated_v2 import *
from models.unet_dilated_v3 import *

from data.data import *
from data.tensorboard_image import *
from data.combined_prediction import *
from data.post_processing import *

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Load arguments from parser
parser = argparser.get_parser()
args = parser.parse_args()
if args.arg_log:
    argparser.write_config_file(args)

# Setting random seeds for tensorflow, numpy and keras
tf.random.set_seed(args.seed)
numpy.random.seed(args.seed)

# Initializing and compiling unet model
if args.model == 'unet':
    model = unet(learning_rate=args.adam_lr)
elif args.model == 'unet_dilated1':
    model = unet_dilated_v1(learning_rate=args.adam_lr)
elif args.model == 'unet_dilated2':
    model = unet_dilated_v2(learning_rate=args.adam_lr)
elif args.model == 'unet_dilated3':
    model = unet_dilated_v3(learning_rate=args.adam_lr)

if args.train_model:
    # Initializing callbacks for training
    callbacks = []

    tensorflow_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-e{}-s{}'.format(args.epochs, args.steps)

    # Initializing logs directory for tensorboard
    log_dir = os.path.join('../logs/fit', tensorflow_dir)
    os.mkdir(log_dir)

    # Initializing tensorboard callback for plots, graph, etc.
    tensorboard_callback = TensorBoard(log_dir=log_dir, write_graph=True)
    callbacks.append(tensorboard_callback)

    # Initializing tensorboard image callback for visualizing validation images each epoch
    tensorboard_image_callback = TensorBoardImage(log_dir=log_dir, validation_path=args.val_path)
    callbacks.append(tensorboard_image_callback)

    # Initialization model checkpoint to store model with best validation loss
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.mkdir(os.path.dirname(args.model_path))

    model_checkpoint_callback = ModelCheckpoint(args.model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    callbacks.append(model_checkpoint_callback)

    # Augmentation parameters for training generator (not validation!)
    data_gen_args = dict(rotation_range=args.rotation_range,
                         width_shift_range=args.width_shift_range,
                         height_shift_range=args.height_shift_range,
                         shear_range=args.shear_range,
                         zoom_range=args.zoom_range,
                         brightness_range=args.brightness_range,
                         horizontal_flip=args.horizontal_flip,
                         vertical_flip=args.vertical_flip,
                         fill_mode=args.fill_mode)

    # Initializing training and validation generators
    train_gen, val_gen = getTrainGenerators(data_gen_args,
                                            train_path=args.train_path, validation_path=args.val_path,
                                            image_folder='images', mask_folder='groundtruth',
                                            target_size=(400, 400), batch_size=args.batch_size, seed=args.seed)

    # Training unet model
    model.fit(train_gen, steps_per_epoch=args.steps, epochs=args.epochs, validation_data=val_gen, validation_steps=args.val_steps, callbacks=callbacks, verbose=1)

# Checking if model weights for best val_loss should be picked for prediction
if args.predict_best:
    # Loading best result
    model.load_weights(args.model_path)

if args.comb_pred:
    # Combining several predictions to one
    predict_combined_results(model, test_path=args.test_path, image_dir='images', result_dir='results',
                             scale_mode=args.scale_mode, gather_mode=args.gather_mode)
else:
    # Initializing test generator
    test_gen = testGenerator(test_path=args.test_path, image_folder='images', target_size=(400, 400))

    # Predicting results on test images
    images = os.listdir(os.path.join(args.test_path, 'images'))
    results = model.predict(test_gen, steps=len(images), verbose=1)

    # Saving result masks of test images
    saveResult(test_path=args.test_path, images=images, results=results)
