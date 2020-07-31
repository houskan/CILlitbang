import numpy as np
import skimage.color
import os


def convert_to_lab(image):
    """This converts an rgb image to lab space.
    :param image: input rgb image (in range [0, 1])
    :return: output lab image (in range [0, 1])
    """
    lab = skimage.color.rgb2lab(image)
    lab_scaled = (lab + [0.0, 128.0, 128.0]) / [100.0, 255.0, 255.0]
    return lab_scaled


def convert_to_hsv(image):
    """This converts an rgb image to hsv space.
    :param image: input rgb image (in range [0, 1])
    :return: output hsv image (in range [0, 1])
    """
    hsv = skimage.color.rgb2hsv(image)
    hsv_scaled = hsv / [360.0, 255.0, 255.0]
    return hsv_scaled


def discretize(result_cont, eps=0.5):
    """This creates a binary image from a continuous one. This is basically the simple thresholding
    approach for our pipeline.
    :param result_cont: continuous mask / probability map
    :param eps: eps is the threshold
    """
    result_disc = result_cont.copy()
    result_disc[result_disc > eps] = 1.0
    result_disc[result_disc <= eps] = 0.0
    return result_disc


def get_path_pairs(path, image_dir, mask_dir):
    """This method pairs images and masks.
    :param path: This is the base path where the image directory and the mask directory are
    :param image_dir: This is the directory where the images are, relative to path
    :param mask_dir: This is the directory where the masks are, relative to path
    :return: list of image/mask pairs
    """
    images = []
    masks = []

    image_files = os.listdir(os.path.join(path, image_dir))
    image_files.sort()

    mask_files = os.listdir(os.path.join(path, mask_dir))
    mask_files.sort()

    for file in image_files:
        images.append(os.path.join(path, image_dir, file))
    for file in mask_files:
        masks.append(os.path.join(path, mask_dir, file))

    result = list(zip(images, masks))
    return result
