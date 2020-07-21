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
from submission.log_submission import *

# Load arguments from parser and saving it if requested
parser = argparser.get_parser()
args = parser.parse_args()
if args.arg_log:
    argparser.write_config_file(args)

# Setting random seeds for tensorflow, numpy and keras
tf.random.set_seed(args.seed)
numpy.random.seed(args.seed)

# Initializing unique submission identifier
date_identifier = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
submission_identifier = date_identifier + '-E{}-S{} {}'.format(args.epochs, args.steps, args.sub_name)

# Initializing and compiling unet model
if args.model == 'unet':
    model = unet(learning_rate=args.adam_lr)
elif args.model == 'unet_dilated1':
    model = unet_dilated_v1(learning_rate=args.adam_lr)
elif args.model == 'unet_dilated2':
    model = unet_dilated_v2(learning_rate=args.adam_lr)
elif args.model == 'unet_dilated3':
    model = unet_dilated_v3(learning_rate=args.adam_lr)
else:
    raise Exception('Unknown model: ' + args.model)

# Checking if model should be trained
if args.train_model:
    # Initializing callbacks for training
    callbacks = []

    # Initializing logs directory for tensorboard
    log_dir = os.path.join('..', 'logs', 'fit', submission_identifier)
    if not os.path.exists(log_dir):
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
    train_gen, val_gen = train_validation_generators(data_gen_args,
                                                     train_path=args.train_path, validation_path=args.val_path,
                                                     image_dir='images', mask_dir='groundtruth',
                                                     target_size=(400, 400), batch_size=args.batch_size, seed=args.seed)

    # Training unet model
    model.fit(train_gen, steps_per_epoch=args.steps, epochs=args.epochs,
              validation_data=val_gen, validation_steps=args.val_steps,
              callbacks=callbacks, verbose=1)

# Checking if model weights for best val_loss should be picked for prediction
if args.predict_best:
    # Loading best result
    model.load_weights(args.model_path)

# Saving images of best prediction model weights in tensorboard by calling callback one more time
if args.predict_best and args.train_model:
    tensorboard_image_callback.on_epoch_end(epoch=args.epochs)

# Predicting results with specific generator, gathering results and saving them depending on
# scale mode, combined prediction boolean, as well as gathering mode
predict_results(model=model, test_path=args.test_path, image_dir='images', result_dir='results',
                scale_mode=args.scale_mode, comb_pred=args.comb_pred, gather_mode=args.gather_mode,
                line_smoothing_mode=args.line_smoothing_mode, apply_hough=args.apply_hough,
                hough_discretize_mode=args.hough_discretize_mode, discretize_mode=args.discretize_mode, region_removal=args.region_removal,
                line_smoothing_R=args.line_smoothing_R, line_smoothing_r=args.line_smoothing_r, line_smoothing_threshold=args.line_smoothing_threshold,
                hough_thresh=args.hough_thresh, hough_min_line_length=args.hough_min_line_length, hough_max_line_gap=args.hough_max_line_gap,
                hough_pixel_up_thresh=args.hough_pixel_up_thresh, hough_eps=args.hough_eps, region_removal_size=args.region_removal_size)

# Checking if submission should be logged and saving all relevant data in unique out submission directory
if args.sub_log:
    log_submission(submission_identifier=submission_identifier, args=args)


