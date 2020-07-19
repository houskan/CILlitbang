import numpy as np
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
import os

from data.helper import *

def apply_transforms(images):
    # Applying rotation and flipping (mirroring) to input images array
    trans_images = np.zeros(images.shape)
    trans_images[0] = images[0]
    trans_images[1] = np.rot90(images[1], 1)
    trans_images[2] = np.rot90(images[2], 2)
    trans_images[3] = np.rot90(images[3], 3)
    trans_images[4] = np.fliplr(images[4])
    trans_images[5] = np.rot90(np.fliplr(images[5]), 1)
    trans_images[6] = np.rot90(np.fliplr(images[6]), 2)
    trans_images[7] = np.rot90(np.fliplr(images[7]), 3)
    return trans_images

def reverse_transforms(results):
    # Reversing rotation and flipping (mirroring) to output results array
    rev_results = np.zeros(results.shape)
    rev_results[0] = results[0]
    rev_results[1] = np.rot90(results[1], 3)
    rev_results[2] = np.rot90(results[2], 2)
    rev_results[3] = np.rot90(results[3], 1)
    rev_results[4] = np.fliplr(results[4])
    rev_results[5] = np.fliplr(np.rot90(results[5], 3))
    rev_results[6] = np.fliplr(np.rot90(results[6], 2))
    rev_results[7] = np.fliplr(np.rot90(results[7], 1))
    return rev_results

def predict_combined(model, img):

    # Constructing all eight rotated and flipped input images
    images = apply_transforms(images=np.broadcast_to(img, (8,) + img.shape))

    # Predicting results for eight rotated and flipped input images
    results = model.predict(images, batch_size=8, verbose=1)

    # Reversing rotation and flipping (mirroring) of resulting continuous output masks
    results = reverse_transforms(results=results)

    return results


def gather_combined(results, mode, thresh):

    if mode == 'avg':

        result_avg_cont = (1.0 / 8.0) * np.sum(results, axis=0)
        result_avg_disc = discretize(result_avg_cont)

        return result_avg_cont, result_avg_disc

    elif mode == 'vote':

        # Converting continuous output results to discrete masks with zero and one values
        results_disc = discretize(results)

        result_vote_sum = np.sum(results_disc, axis=0)
        result_vote_avg = (1.0 / 8.0) * result_vote_sum
        result_vote_disc = result_vote_sum.copy()
        result_vote_disc[result_vote_disc < thresh] = 0.0
        result_vote_disc[result_vote_disc >= thresh] = 1.0

        return result_vote_avg, result_vote_disc

    else:
        raise Exception('Unknown gathering mode: ' + mode)


def predict_window(model, img, target_size, window_stride, gather_mode, vote_thresh):

    mask_cont = np.zeros(img.shape)
    mask_disc = np.zeros(img.shape)
    overlaps = np.zeros(img.shape)

    for i in range(0, img.shape[0] - target_size[0] + 1, window_stride[0]):
        for j in range(0, img.shape[1] - target_size[1] + 1, window_stride[1]):

            # Getting window image with target size
            img_window = img[i:i+target_size[0], j:j+target_size[1], :]

            # Predicting results for all transformations of window image
            results = predict_combined(model=model, img=img_window)

            # Gathering results back to one continuous and discrete window with specific mode
            mask_cont_window, mask_disc_window = gather_combined(results=results, mode=gather_mode, thresh=vote_thresh)

            overlaps[i:i+target_size[0], j:j+target_size[1], :] += 1.0
            mask_cont[i:i+target_size[0], j:j+target_size[1], :] += mask_cont_window
            mask_disc[i:i+target_size[0], j:j+target_size[1], :] += mask_disc_window

    mask_cont = mask_cont / overlaps
    mask_disc = discretize(mask_cont)

    return mask_cont, mask_disc

def predict_resize(model, img, target_size, gather_mode, vote_thresh):

    # Saving original size of input (without number of channels)
    original_size = img.shape[:-1]

    img = trans.resize(img, target_size)

    results = predict_combined(model=model, img=img)

    # Resizing resulting output masks to original size
    resized_results = np.zeros((results.shape[0],) + original_size + (1,))
    for i in range(results.shape[0]):
        resized_results[i] = trans.resize(results[i], original_size + (1,))

    mask_cont, mask_disc = gather_combined(results=resized_results, mode=gather_mode, thresh=vote_thresh)

    return mask_cont, mask_disc

def predict_combined_results(model, test_path, image_dir, result_dir, scale_mode='resize', gather_mode='avg'):

    for file in os.listdir(os.path.join(test_path, image_dir)):

        # Getting image name (without extension)
        img_name = file.split('.')[-2]

        # Reading image and rescaling to float range [0, 1], if necessary
        img = io.imread(os.path.join(test_path, image_dir, file), as_gray=False)
        if np.max(img) > 1.0:
            img = img / 255.0

        if scale_mode == 'resize':
            mask_cont, mask_disc = predict_resize(model=model, img=img, target_size=(400, 400),
                                                  gather_mode=gather_mode, vote_thresh=5)
        elif scale_mode == 'window':
            mask_cont, mask_disc = predict_window(model=model, img=img, target_size=(400, 400), window_stride=(208, 208),
                                                  gather_mode=gather_mode, vote_thresh=5)
        else:
            raise Exception('Unknown scale mode: ' + scale_mode)

        # Saving discrete and continuous masks in respective folders in result directory
        io.imsave(os.path.join(test_path, result_dir, 'discrete', img_name + '.png'), img_as_ubyte(mask_disc))
        io.imsave(os.path.join(test_path, result_dir, 'continuous', img_name + '.png'), img_as_ubyte(mask_cont))

        # Also saving discrete and continuous masks in results directory (for easier comparison)
        io.imsave(os.path.join(test_path, result_dir, img_name + '_disc.png'), img_as_ubyte(mask_disc))
        io.imsave(os.path.join(test_path, result_dir, img_name + '_cont.png'), img_as_ubyte(mask_cont))


def saveResult(test_path, images, results):

    # Initializing two list for test image mask result file path names (first discrete, second continuous)
    #resultNames = list(map(lambda x: os.path.join(test_path, 'results', x), images))
    resultNames = list(map(lambda x: os.path.join(test_path, 'results', '{0}_discS.{1}'.format(*x.rsplit('.', 1))), images))
    resultNamesCont = list(map(lambda x: os.path.join(test_path, 'results', '{0}_contS.{1}'.format(*x.rsplit('.', 1))), images))

    # Iterating through all result masks
    for i, item in enumerate(results):

        # Initializing new mask image and discretizing it with threshold=0.5 to either 0 or 1
        mask = item.copy()
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        # Resizing discrete mask back to original image size (608, 608) and saving it to result file
        mask = trans.resize(mask, (608, 608))
        io.imsave(resultNames[i], img_as_ubyte(mask))

        # Copying raw unet output, resizing this continuous mask back to original image size (608, 608),
        # and saving it to result file
        mask_cont = item.copy()
        mask_cont = trans.resize(mask_cont, (608, 608))
        io.imsave(resultNamesCont[i], img_as_ubyte(mask_cont))
