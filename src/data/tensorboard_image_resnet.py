import tensorflow as tf
from tensorflow import keras
#import keras
import cv2
import numpy as np

class TensorBoardImageResnet(keras.callbacks.Callback):
    def __init__(self, log_dir, validation_pairs):
        super().__init__()
        self.log_dir = log_dir
        self.validation_pairs = validation_pairs

    def on_epoch_end(self, epoch, logs={}):
        # TODO: (Sebastian) Major cleanup!!!
        # Iterating through all validation image path pairs
        for index, item in enumerate(self.validation_pairs):
            # Getting input and output height and width of model
            input_height = self.model.input_height
            input_width = self.model.input_width
            output_height = self.model.output_height
            output_width = self.model.output_width

            # Getting colors and number of classes for mask data
            colors = [(0, 0, 0), (255, 255, 255)]
            n_classes = 2

            # Getting original input and mask image paths
            img_path, mask_path = item

            # Reading original input image and mask to get (height, width, 3) numpy arrays
            original_img = cv2.imread(img_path)
            groundtruth_mask = cv2.imread(mask_path)

            # TODO: (Sebastian) cleanup...
            test_img = cv2.resize(original_img, (input_width, input_height))
            test_img = test_img.astype(np.float32)
            test_img[:, :, 0] -= 103.939
            test_img[:, :, 1] -= 116.779
            test_img[:, :, 2] -= 123.68
            test_img = test_img[:, :, ::-1]

            # Predicting mask of test image with model
            result_mask = self.model.predict(np.array([test_img]))[0]

            # TODO: (Sebastian) cleanup...
            # Converting mask back to (400, 400, 1) mask in uint8
            result_mask = result_mask.reshape((output_height, output_width, n_classes)).argmax(axis=2)
            seg_img = np.zeros((output_height, output_width, 3))
            for c in range(n_classes):
                seg_arr_c = result_mask[:, :] == c
                seg_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')
            result_mask = cv2.resize(seg_img, (400, 400))

            # Appending extra dimension to the front for all images (tensorboard image requirements)
            result_mask = np.reshape(result_mask, (1,) + result_mask.shape)
            groundtruth_mask = np.reshape(groundtruth_mask, (1,) + groundtruth_mask.shape)
            original_img = np.reshape(original_img, (1,) + original_img.shape)

            # Creating tensorflow summary file writer at log directory to save images
            file_writer = tf.summary.create_file_writer(self.log_dir)

            with file_writer.as_default():
                # Adding mask prediction for current epoch to mask prediction slider
                tf.summary.image('result/val_img%d/mask_prediction' % index, result_mask, step=epoch)

                # Adding groundtruth mask and original input image to ith validation image in first epoch as comparison
                if (epoch==0):
                    tf.summary.image('result/val_img%d/mask_groundtruth' % index, groundtruth_mask, step=0)
                    tf.summary.image('result/val_img%d/input' % index, original_img, step=0)
