import numpy as np
from skimage import measure, filters
import cv2
import scipy
import maxflow
import skimage
import argparser

from data.helper import *


def postprocess(img, mask_cont, mask_disc, args):
#               line_smoothing_mode, apply_hough, hough_discretize_mode, discretize_mode,
#               region_removal, region_removal_size, line_smoothing_R, line_smoothing_r, line_smoothing_threshold,
#               hough_thresh, hough_min_line_length, hough_max_line_gap, hough_pixel_up_thresh, hough_eps,
#               hough_discretize_thresh):
    """
    This method implements the complete post-processing pipeline. It enables all reasonable
    combinations of the individual steps. 
    :param img: RGB image
    :param mask_cont: Continous mask / Probability map / Output of the neural network
    :param other parameters: see argparser
    :return: Postprocessed binary segmentation mask
    """

    if args.line_smoothing_mode == 'beforeHough' or args.line_smoothing_mode == 'both':
        mask_cont = line_smoothing(mask_cont, R=args.line_smoothing_R, r=args.line_smoothing_r,
                                   threshold=args.line_smoothing_threshold)

    # Apply Hough Transform
    if args.apply_hough:
        # Apply Hough dependent on discretize functio
        if args.hough_discretize_mode == 'discretize':
            mask_cont = hough_pipeline(mask_cont, np.ones((3, 3), np.uint8),
                                       lambda x: discretize(x, args.hough_discretize_thresh),
                                       hough_thresh=args.hough_thresh,
                                       min_line_length=args.hough_min_line_length,
                                       max_line_gap=args.hough_max_line_gap,
                                       pixel_up_thresh=args.hough_pixel_up_thresh,
                                       eps=args.hough_eps)
        elif args.hough_discretize_mode == 'graphcut':
            mask_cont = hough_pipeline(mask_cont, np.ones((3, 3), np.uint8), lambda x: graph_cut(x, img),
                                       hough_thresh=args.hough_thresh,
                                       min_line_length=args.hough_min_line_length,
                                       max_line_gap=args.hough_max_line_gap,
                                       pixel_up_thresh=args.hough_pixel_up_thresh,
                                       eps=args.hough_eps)
        else:
            raise Exception('Unknown discretize mode for Hough postprocessing: ' + args.hough_discretize_mode)

    # Smooth Lines after Hough post-processing
    if args.line_smoothing_mode == 'afterHough' or args.line_smoothing_mode == 'both':
        mask_cont = line_smoothing(mask_cont, R=args.line_smoothing_R, r=args.line_smoothing_r,
                                   threshold=args.line_smoothing_threshold)

    # Discretize the probability map
    if args.discretize_mode == 'discretize':
        discretize_function = discretize
    elif args.discretize_mode == 'graphcut':
        discretize_function = lambda x: graph_cut(x, img)
    else:
        raise Exception('Unknown discretize mode for final discretization: ' + args.discretize_mode)
    mask_disc = discretize_function(mask_cont)

    # Remove Small Regions
    if args.region_removal:
        mask_disc = remove_small_regions(mask_disc, no_pixels=args.region_removal_size)

    return mask_cont, mask_disc


def remove_small_regions(mask, no_pixels=256):
    """Cleans up a discrete prediction by removing small regions
    from the binary segmentation mask
    :param mask: binary segmentation mask, not getting modified
    :param no_pixels: minimum region size
    :return: new cleaned up binary segmentation msk
    """
    mres = mask.copy()
    # detect and measure regions
    all_labels = measure.label(mres)
    mres = measure.label(mres, background=0)
    uniq = np.unique(mres, return_counts=True)
    labs = uniq[0]
    cnts = uniq[1]

    # remove set small regions to 0
    for i, lab in enumerate(labs):
        if cnts[i] < no_pixels:
            mres[np.where(mres == lab)] = 0

    mres[np.where(mres != 0)] = 1
    return mres.astype('float32')


def get_hough_lines(mask, threshold=100, min_line_length=1, max_line_gap=500):
    """Returns numpy array containing the number of hough lines passing through it.
    :param mask: binary mask, Two dimensions only
    :param threshold: see cv2.HoughLinesP
    :param min_line_length: see cv2.HoughLinesP
    :param max_line_gap: see cv2.HoughLinesP
    :return: numpy array containing hough line count per pixel
    """
    gray = (mask * 255).astype('uint8')
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    hough_lines = np.zeros(gray.shape)
    if not lines is None:
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                one_hough = np.zeros(gray.shape)
                cv2.line(one_hough, (x1, y1), (x2, y2), 1., 1)
                hough_lines = hough_lines + one_hough

    return hough_lines


def hough_update_mask(mask, hough_lines, kernel, thresh=1, eps=0.2):
    """Updates the mask by increasing probabilities using hough_lines.
    :param mask: continous mask / probability map, not getting modified
    :param hough_lines: result obtained from  get_hough_lines
    :param kernel: for morphological closing applied on hough_lines for smoothing them
        If kernel is not None, morphological closing is not applied
        good kernel example np.ones((3,3),np.uint8)
    :param thresh: How many lines need to pass through a pixel at least
    :param eps:  Which constant factor should be added to chosen pixels
    :return: new updated continous mask / probability map
    """
    updated_mask = mask.copy()
    hough_lines_c = hough_lines.copy()
    if not kernel is None:
        hough_lines_c = cv2.morphologyEx(hough_lines_c, cv2.MORPH_CLOSE, kernel)
    updated_mask = updated_mask + eps * (hough_lines_c >= thresh)
    return updated_mask


def hough_pipeline(mask, kernel, discretize_func, hough_thresh=100, min_line_length=1,
                   max_line_gap=500, pixel_up_thresh=1, eps=0.2):
    """This method performs the complete update of probability
    maps using the hough transform to detect road segments
    with lower probability.
    :param mask: continous mask, not getting modified
    :param kernel: for morphological closing applied on hough_lines for smoothing them
        If kernel is not None, morphological closing is not applied
        good kernel example np.ones((3,3),np.uint8)
    :param hough_thresh: see cv2.HoughLinesP
    :param min_line_length: see cv2.HoughLinesP
    :param max_line_gap: see cv2.HoughLinesP
    :param thresh: How many lines need to pass through a pixel at least
    :param eps: Which constant factor should be added to chosen pixels
    :return: continous mask
    """
    disc_mask = discretize_func(mask)
    hough_lines = get_hough_lines(disc_mask, threshold=hough_thresh, min_line_length=min_line_length,
                                  max_line_gap=max_line_gap)
    updated_mask = hough_update_mask(mask, hough_lines, kernel, thresh=pixel_up_thresh, eps=eps)
    return np.clip(updated_mask, a_min=0.0, a_max=1.0)


def adjust_data_for_graphcut(img):
    """
    Some possible way to preprocess the colored test image for graph cut
    :param img: image for preprocessing
    :return: preprocessed image
    """
    img = skimage.color.rgb2lab(img)
    img = filters.gaussian(img, sigma=1, multichannel=True)
    if np.max(img) > 1.0:
        img = img / 255.0
    return img


def graph_cut(prediction, img, lambda_=1, sigma=3):
    """
    This method requires The PyMaxflow library. It takes the continuous prediction plus the original image to
    determine pixel assignment to either background or road. This is done by minimizing an energy function by finding a
    min cut (e.g. max flow)
    :param prediction: continuous prediction (grey scale image)
    :param img: original colored test image
    :return: discretized mask
    """

    img = adjust_data_for_graphcut(img)

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(prediction.shape)

    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])

    img_right = np.roll(img, 1, axis=2)
    weights = img - img_right
    weights = np.multiply(weights, weights)
    weights = weights[:, :, 0] + weights[:, :, 1] + weights[:, :, 2]

    weights = weights / (2 * sigma * sigma)
    weights = np.exp(-weights)

    g.add_grid_edges(nodeids, weights=lambda_ * weights, structure=structure, symmetric=True)

    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])

    img_down = np.rot90(img)
    img_down = np.roll(img_down, 1, axis=2)
    img_down = np.rot90(img_down, k=-1)
    weights = img - img_down
    weights = np.multiply(weights, weights)
    weights = weights[:, :, 0] + weights[:, :, 1] + weights[:, :, 2]

    weights = weights / (2 * sigma * sigma)
    weights = np.exp(-weights)

    g.add_grid_edges(nodeids, weights=lambda_ * weights, structure=structure, symmetric=True)

    g.add_grid_tedges(nodeids, prediction, 1 - prediction)

    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    result = np.int_(np.logical_not(sgm))
    result *= 255
    return result


def line_smoothing(prediction, R=20, r=3, threshold=0.25):
    """
    This method implements simple line smoothing
    :param prediction: continuous mask
    :param R: how far up an down is looked to find maximum value to assign to current pixel
    :param r: how far up and down there needs to be at least 1 pixel value higher than threshold
    :param threshold:
    :return: continuous mask with certain pixel probabilities increased
    """
    footprint_0 = np.zeros((2 * R + 1, 2 * R + 1))
    footprint_0[R, R:2 * R + 1] = 1
    footprint_45 = np.zeros((2 * R + 1, 2 * R + 1))
    for i in range(R + 1):
        footprint_45[i, i] = 1

    footprint_90 = np.rot90(footprint_0)
    footprint_135 = np.rot90(footprint_45)
    footprint_180 = np.rot90(footprint_90)
    footprint_225 = np.rot90(footprint_135)
    footprint_270 = np.rot90(footprint_180)
    footprint_315 = np.rot90(footprint_225)

    footprints_R = []
    footprints_R.append(footprint_0)
    footprints_R.append(footprint_45)
    footprints_R.append(footprint_90)
    footprints_R.append(footprint_135)
    footprints_R.append(footprint_180)
    footprints_R.append(footprint_225)
    footprints_R.append(footprint_270)
    footprints_R.append(footprint_315)

    footprint_0 = np.zeros((2 * r + 1, 2 * r + 1))
    footprint_0[r, r:2 * r + 1] = 1

    footprint_45 = np.zeros((2 * r + 1, 2 * r + 1))
    for i in range(r + 1):
        footprint_45[i, i] = 1

    footprint_90 = np.rot90(footprint_0)
    footprint_135 = np.rot90(footprint_45)
    footprint_180 = np.rot90(footprint_90)
    footprint_225 = np.rot90(footprint_135)
    footprint_270 = np.rot90(footprint_180)
    footprint_315 = np.rot90(footprint_225)

    footprints_r = []
    footprints_r.append(footprint_0)
    footprints_r.append(footprint_45)
    footprints_r.append(footprint_90)
    footprints_r.append(footprint_135)
    footprints_r.append(footprint_180)
    footprints_r.append(footprint_225)
    footprints_r.append(footprint_270)
    footprints_r.append(footprint_315)

    results = []
    for i in range(4):
        S_up_filter = footprints_R[i]
        S_down_filter = footprints_R[i + 4]
        T_up_filter = footprints_r[i]
        T_down_filter = footprints_r[i + 4]

        S_up = scipy.ndimage.maximum_filter(prediction, footprint=S_up_filter)
        S_down = scipy.ndimage.maximum_filter(prediction, footprint=S_down_filter)
        T_up = scipy.ndimage.maximum_filter(prediction, footprint=T_up_filter)
        T_down = scipy.ndimage.maximum_filter(prediction, footprint=T_down_filter)

        T_up = T_up > threshold
        T_down = T_down > threshold
        s_up = np.multiply(S_up, T_up)
        s_down = np.multiply(S_down, T_down)

        result = np.maximum(prediction, np.minimum(s_up, s_down))
        results.append(result)

    smoothed = np.maximum(results[0], results[1], np.maximum(results[2], results[3]))
    return smoothed
