import numpy as np
import skimage.color
import os

def discretize(result_cont, eps=0.5):
    result_disc = result_cont.copy()
    result_disc[result_disc > eps] = 1.0
    result_disc[result_disc <= eps] = 0.0
    return result_disc

def get_path_pairs(path, image_dir, mask_dir):
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
