import numpy as np
import skimage.color
import os

def convertToLab(image):
    lab = skimage.color.rgb2lab(image)
    lab_scaled = (lab + [0.0, 128.0, 128.0]) / [100.0, 255.0, 255.0]
    return lab_scaled

def convertToHSV(image):
    hsv = skimage.color.rgb2hsv(image)
    hsv_scaled = hsv / [360.0, 255.0, 255.0]
    return hsv_scaled

def discretize(result_cont, eps=0.5):
    '''This creates a binary image from a continous one. This is basically the simple thresholding
    approach for our pipeline.
    :param result_cont: continous mask / probability map
    :param eps: eps is the threshold
    '''
    result_disc = result_cont.copy()
    result_disc[result_disc > eps] = 1.0
    result_disc[result_disc <= eps] = 0.0
    return result_disc

def get_path_pairs(path, image_dir, mask_dir):
    '''This method pairs images and masks. 
    :param path: This is the base path where the image directory and the mask directory are
    :param image_dir: This is the directory where the images are, relative to path
    :param mask_dir: This is the directory where the masks are, relative to path
    :return: list of image/mask pairs
    '''
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
