import numpy as np
import tensorflow as tf
from tensorflow import keras

import skimage.io as io
import skimage.color
# import skimage.transform as trans
import os

from data.helper import *

'''
This file contains the callbacks for tensorboard
'''
class TensorBoardImage(keras.callbacks.Callback):

    def __init__(self, log_dir, validation_path, image_dir='images', mask_dir='groundtruth', target_size=(400, 400)):
        super().__init__()
        self.log_dir = log_dir
        self.validation_pairs = get_path_pairs(validation_path, image_dir, mask_dir)
        self.target_size = target_size

    def on_epoch_end(self, epoch, logs={}):

        # Iterating through all validation image path pairs
        for index, (img_path, mask_path) in enumerate(self.validation_pairs):

            # Reading original image input image
            input_img = io.imread(img_path, as_gray=False)
            if np.max(input_img) > 1.0:
                input_img = input_img / 255.0

            # Predicting result mask of input image with model
            result = self.model.predict(np.array([input_img]))[0]

            # Initializing two result mask versions: one continuous and one discrete with threshold=0.5
            result_mask_cont = result.copy()
            result_mask_disc = discretize(result)

            # Converting to result mask version back to range uint8 [0, 255] (tensorboard image requirements)
            result_mask_cont = (result_mask_cont * 255.0).astype('uint8')
            result_mask_disc = (result_mask_disc * 255.0).astype('uint8')

            # Converting mask dimensions from greyscale to rgb: (400, 400, 1) -> (400, 400, 3) (tensorboard image requirements)
            result_mask_cont = np.repeat(result_mask_cont, 3, axis=2)
            result_mask_disc = np.repeat(result_mask_disc, 3, axis=2)

            # Appending extra dimension to the front for all images (tensorboard image requirements)
            result_mask_cont = np.reshape(result_mask_cont, (1,) + result_mask_cont.shape)
            result_mask_disc = np.reshape(result_mask_disc, (1,) + result_mask_disc.shape)

            # Creating tensorflow summary file writer at log directory to save images
            file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'images'))

            with file_writer.as_default():

                # Adding mask prediction (discrete and continuous) for current epoch to mask prediction slider
                tf.summary.image('result/val_img%d/mask_prediction (continuous)' % index, result_mask_cont, step=epoch)
                tf.summary.image('result/val_img%d/mask_prediction (discrete)' % index, result_mask_disc, step=epoch)

                # Adding original input image and groundtruth mask in first epoch for comparison
                if epoch == 0:
                    # Reading original image and groundtruth mask for comparisons
                    original_img = io.imread(img_path, as_gray=False)
                    groundtruth_mask = io.imread(mask_path, as_gray=False)

                    # Adding 3 RGB channels, in case skimage only loads grayscale image with shape (400, 400)
                    if len(groundtruth_mask.shape) == 2:
                        groundtruth_mask = skimage.color.gray2rgb(groundtruth_mask)

                    # Appending extra dimension to the front (tensorboard image requirements)
                    original_img = np.reshape(original_img, (1,) + original_img.shape)
                    groundtruth_mask = np.reshape(groundtruth_mask, (1,) + groundtruth_mask.shape)

                    # Adding input image and mask groundtruth for comparison
                    tf.summary.image('result/val_img%d/input_img' % index, original_img, step=0)
                    tf.summary.image('result/val_img%d/mask_groundtruth' % index, groundtruth_mask, step=0)
