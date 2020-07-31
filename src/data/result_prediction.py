import numpy as np
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
import argparser
import os

from data.helper import *
from data.post_processing import *


def apply_transforms(images):
    """This method takes images and applies multiple unique operations on them
    :param images: array of images
    :return: transformed images
    """
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
    """This method reverts the transforms that apply_transforms did. We used this for reverting the
    multiple predictions back to the original non-transformed setting.
    :param results: continuous masks / probability map
    :return: Back-Transformed results
    """
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


def gather_combined(results, mode, thresh):
    """This method impleemnts the combination of multiple prediction into one. Either by voting or by averaging
    :param results: Array of continuous probability masks
    :param mode: Choose between averaging(avg) and voting (vote)
    :return: resulting mask continuous and resulting mask binarized
    """
    # Checking which gather mode should be used (averaging continuous masks or threshold voting discrete masks)
    if mode == 'avg':
        # Computing average continuous mask and discretizing this averaged result
        result_avg_cont = (1.0 / 8.0) * np.sum(results, axis=0)
        result_avg_disc = discretize(result_avg_cont)
        return result_avg_cont, result_avg_disc
    elif mode == 'vote':
        # Discretizing each of the stacked continuous output results and summing up these discrete masks
        results_disc = discretize(results)
        result_vote_sum = np.sum(results_disc, axis=0)
        # Creating pseudo continuous mask by normalizing/averaging this summed up result
        result_vote_avg = (1.0 / 8.0) * result_vote_sum
        # Threshold voting discrete mask with everything < thresh being no road and everything >= thresh being a road
        result_vote_disc = result_vote_sum.copy()
        result_vote_disc[result_vote_disc < thresh] = 0.0
        result_vote_disc[result_vote_disc >= thresh] = 1.0
        return result_vote_avg, result_vote_disc
    else:
        raise Exception('Unknown gathering mode: ' + mode)


def test_generator(test_path, image_dir='images', target_size=(400, 400),
                   scale_mode='resize', window_stride=(208, 208), gather_mode='avg'):
    """This provides the generator for the test images
    :param test_path: This path is where the image directory is
    :param image_dir: relative path to test_path, this directory contains the images
    :param target_size: What size should the RGB images have
    :param scale_mode: Describes if you want rescale images or apply a sliding window mechanism
    :param window_stride: Parameter for the sliding window mechanism
    :param gather_mode: Describes what kind of mode you want to use when combining results (for transformations)
    """
    # Iterating through all images in test set
    for file in os.listdir(os.path.join(test_path, image_dir)):
        # Reading input test image image and normalizing it to range [0, 1],
        img = io.imread(os.path.join(test_path, image_dir, file), as_gray=False)
        if np.max(img) > 1.0:
            img = img / 255.0

        # Getting original size of input test image
        original_size = img.shape[0:2]

        # Checking which scale mode should be used (resizing image to fit target size or sliding window)
        if scale_mode == 'resize':
            # Resizing image to target size
            img = trans.resize(img, target_size + (3,))
            # Checking combined prediction should yielding all eight transforms or only single image
            if gather_mode == 'avg' or gather_mode == 'vote':
                yield apply_transforms(images=np.broadcast_to(img, (8,) + img.shape))
            else:
                img = np.reshape(img, (1,) + img.shape)
                yield img
        elif scale_mode == 'window':
            # Iterating through input image with window stride in x and y direction (sliding window)
            for i in range(0, original_size[0] - target_size[0] + 1, window_stride[0]):
                for j in range(0, original_size[1] - target_size[1] + 1, window_stride[1]):
                    # Getting window image with target size
                    img_window = img[i:i+target_size[0], j:j+target_size[1], :]
                    # Checking combined prediction should yielding all eight transforms or only single image
                    if gather_mode == 'avg' or gather_mode == 'vote':
                        yield apply_transforms(images=np.broadcast_to(img_window, (8,) + img_window.shape))
                    else:
                        img_window = np.reshape(img_window, (1,) + img_window.shape)
                        yield img_window
        else:
            raise Exception('Unknown scale mode: ' + scale_mode)


def save_results(results, test_path, image_dir, result_dir, args, target_size=(400, 400), window_stride=(208, 208)):
    """This method saves results according to the parameters
    :param results: Array of continuous masks
    :param test_path: Path of the test images
    :param image_dir: relative path to test_path, this directory hosts the images
    :param result_dir: relative path to test_path, this directory hosts the results
    :param others: see argparser for help
    """

    # Initializing index to keep track of where we are in results tensor abd batch size stride
    index = 0
    batch_size = 8 if (args.gather_mode == 'avg' or args.gather_mode == 'vote') else 1

    # Iterating through all images in test set
    for file in os.listdir(os.path.join(test_path, image_dir)):

        # Reading input test image and getting size of it
        img = io.imread(os.path.join(test_path, image_dir, file), as_gray=False)
        original_size = img.shape[0:2]

        # Getting image name (without extension)
        img_name = file.split('.')[-2]

        # Checking which scale mode should be used (resizing image to fit target size or sliding window)
        if args.scale_mode == 'resize':
            # Checking if combined prediction should be applied
            if args.gather_mode == 'avg' or args.gather_mode == 'vote':
                # Reversing transformations of results
                res = reverse_transforms(results=results[index:index + 8])
                # Resizing resulting output masks to original size
                resized_results = np.zeros((8,) + original_size + (1,))
                for i in range(8):
                    resized_results[i] = trans.resize(res[i], original_size + (1,))
                mask_cont, mask_disc = gather_combined(results=resized_results,
                                                       mode=args.gather_mode,
                                                       thresh=args.vote_thresh)
            else:
                # Resizing resulting output mask to original size
                resized_result = trans.resize(results[index], original_size + (1,))
                mask_cont = resized_result
                mask_disc = discretize(mask_cont)
            # Updating results index by adding batch size
            index += batch_size
        elif args.scale_mode == 'window':
            # Initializing bookkeeping 2d arrays for access counts, as well as continuous and discrete accumulators
            counts = np.zeros(img.shape)
            mask_cont = np.zeros(img.shape)
            mask_disc = np.zeros(img.shape)
            # Iterating through input image with window stride in x and y direction (sliding window)
            for i in range(0, original_size[0] - target_size[0] + 1, window_stride[0]):
                for j in range(0, original_size[1] - target_size[1] + 1, window_stride[1]):
                    # Checking if combined prediction should be applied
                    if args.gather_mode == 'avg' or args.gather_mode == 'vote':
                        # Reversing transformations of results
                        res = reverse_transforms(results=results[index:index+8])
                        # Gathering results back to one continuous and discrete window with specific mode
                        mask_window_cont, mask_window_disc = gather_combined(results=res,
                                                                             mode=args.gather_mode,
                                                                             thresh=args.vote_thresh)
                    else:
                        mask_window_cont = results[index]
                        mask_window_disc = discretize(mask_window_cont)
                    # Updating results index by adding batch size
                    index += batch_size
                    # Bookkeeping of elements accessed (counts), as well as original sized continuous and discrete masks
                    counts[i:i+target_size[0], j:j+target_size[1], :] += 1.0
                    mask_cont[i:i+target_size[0], j:j+target_size[1], :] += mask_window_cont
                    mask_disc[i:i+target_size[0], j:j+target_size[1], :] += mask_window_disc
            # Averaging of continuous mask and discretizing it
            mask_cont = mask_cont / counts
            mask_disc = discretize(mask_cont)
        else:
            raise Exception('Unknown scale mode: ' + args.scale_mode)

        mask_cont = mask_cont[:, :, 0]
        mask_disc = mask_disc[:, :, 0]

        print('Saving discrete and continuous mask of image: ' + file)

        # Initializing and possibly also creating directories for discrete and continuous results
        result_path = os.path.join(test_path, result_dir)
        if not os.path.exists(os.path.join(test_path, result_dir)):
            os.mkdir(result_path)
        disc_path = os.path.join(test_path, result_dir, 'discrete')
        if not os.path.exists(disc_path):
            os.mkdir(disc_path)
        cont_path = os.path.join(test_path, result_dir, 'continuous')
        if not os.path.exists(cont_path):
            os.mkdir(cont_path)

        # Saving discrete and continuous masks in respective folders in result directory
        io.imsave(os.path.join(disc_path, img_name + '.png'), img_as_ubyte(mask_disc))
        io.imsave(os.path.join(cont_path, img_name + '.png'), img_as_ubyte(mask_cont))

        # Post processing results with hough transform, graphcut, region removal (and more)
        mask_cont, mask_disc = postprocess(img=img, mask_cont=mask_cont, mask_disc=mask_disc, args=args)

        # Save post processed
        disc_path = os.path.join(test_path, result_dir, 'discrete_post')
        if not os.path.exists(disc_path):
            os.mkdir(disc_path)
        cont_path = os.path.join(test_path, result_dir, 'continuous_post')
        if not os.path.exists(cont_path):
            os.mkdir(cont_path)

        io.imsave(os.path.join(disc_path, img_name + '.png'), img_as_ubyte(mask_disc))
        io.imsave(os.path.join(cont_path, img_name + '.png'), img_as_ubyte(mask_cont))


def predict_results(model, test_path, image_dir, result_dir, args, target_size=(400, 400), window_stride=(208, 208)):
    """This method predicts and saves results
    :param model: Tensorflow model that will predict
    :param results: Array of continuous masks
    :param test_path: Path of the test images
    :param image_dir: relative path to test_path, this directory hosts the images
    :param result_dir: relative path to test_path, this directory hosts the results
    :param args: all other parameters necessary for gathering mode, scaling mode and post processing
    (see argparser for help)
    :param target_size: target size of image
    :window_stride: stride if scaling mode is sliding window
    """

    # Initializing combined test generator (different number of input images depending on scale mode and window stride)
    test_gen = test_generator(test_path=test_path, image_dir=image_dir, target_size=target_size,
                              scale_mode=args.scale_mode, window_stride=window_stride, gather_mode=args.gather_mode)

    # Setting batch and validation steps for prediction with generator
    batch_size = 8 if (args.gather_mode == 'avg' or args.gather_mode == 'vote') else 1
    val_steps = len(os.listdir(os.path.join(test_path, image_dir)))
    if args.scale_mode == 'window':
        # Adding multiplier factor for window images per input test image
        val_steps *= ((608 - target_size[0]) // window_stride[0] + 1) * ((608 - target_size[1]) // window_stride[1] + 1)

    # Predicting results with combined test generator for all test images and with batch size one
    results = model.predict(test_gen, steps=val_steps, batch_size=batch_size, verbose=1)

    # Gathering raw results (in case of sliding window) and saving results
    save_results(results=results, test_path=test_path, image_dir=image_dir, result_dir=result_dir, args=args,
                 target_size=target_size, window_stride=window_stride)
