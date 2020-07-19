import tensorflow as tf
from tensorflow import keras
import numpy as np

import cv2
import skimage.io as io
#import skimage.transform as trans
import os

from data.helper import *

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, log_dir, validation_pairs, target_size=(400, 400)):
        super().__init__()
        self.log_dir = log_dir
        self.validation_pairs = validation_pairs
        self.target_size = target_size

    def on_epoch_end(self, epoch, logs={}):
        # TODO: (Sebastian) Cleanup!
        # Iterating through all validation image path pairs
        for index, item in enumerate(self.validation_pairs):

            # Getting original input and mask image paths
            img_path, mask_path = item

            # Reading original input image and mask to get (height, width, 3) numpy arrays
            original_img = cv2.imread(img_path)
            groundtruth_mask = cv2.imread(mask_path)

            # Reading original image with skimage io
            # (for some reason cv2 does not work???)
            input_img = io.imread(img_path, as_gray=False)
            if np.max(input_img) > 1.0:
                input_img = input_img / 255.0
            #input_img = original_img.copy()

            # Predicting mask of test image with model
            result = self.model.predict(np.array([input_img]))[0]

            # Initializing two result mask versions: one continuous and one discrete with threshold=0.5
            result_mask_cont = result.copy()
            result_mask = result.copy()
            result_mask[result_mask > 0.5] = 1.0
            result_mask[result_mask <= 0.5] = 0.0

            # Converting to result mask version back to range uint8 [0, 255] (tensorboard image requirements)
            result_mask_cont = (result_mask_cont * 255.0).astype('uint8')
            result_mask = (result_mask * 255.0).astype('uint8')

            # Converting mask dimensions from greyscale to rgb: (400, 400, 1) -> (400, 400, 3) (tensorboard image requirements)
            result_mask_cont = np.repeat(result_mask_cont, 3, axis=2)
            result_mask = np.repeat(result_mask, 3, axis=2)

            # Appending extra dimension to the front for all images (tensorboard image requirements)
            result_mask = np.reshape(result_mask, (1,) + result_mask.shape)
            result_mask_cont = np.reshape(result_mask_cont, (1,) + result_mask_cont.shape)
            groundtruth_mask = np.reshape(groundtruth_mask, (1,) + groundtruth_mask.shape)
            original_img = np.reshape(original_img, (1,) + original_img.shape)

            # Creating tensorflow summary file writer at log directory to save images
            file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'images'))

            with file_writer.as_default():
                # Adding mask prediction (discrete and continuous) for current epoch to mask prediction slider
                tf.summary.image('result/val_img%d/mask_prediction (thresh=0.5)' % index, result_mask, step=epoch)
                tf.summary.image('result/val_img%d/mask_prediction (continuous)' % index, result_mask_cont, step=epoch)

                # Adding groundtruth mask and original input image to ith validation image in first epoch as comparison
                if epoch == 0:
                    tf.summary.image('result/val_img%d/mask_groundtruth' % index, groundtruth_mask, step=0)
                    tf.summary.image('result/val_img%d/input' % index, original_img, step=0)
