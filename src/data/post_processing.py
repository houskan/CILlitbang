import numpy as np
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte, measure
import cv2

from data.helper import *

import os

def remove_small_regions(mask, no_pixels=256):
    """Cleans up a discrete prediction by removing small regions
    from the discrete mask
    mask -- discretized mask, not getting modified
    no_pixels -- minimum region size
    return -- new cleaned up discrete mask
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
            mres[np.where(mres==lab)] = 0

    mres[np.where(mres != 0)] = 1
    return mres.astype('float32')

def get_hough_lines(mask, threshold=100, min_line_length=1, max_line_gap=500):
    """Returns numpy array containing the number of hough lines passing through it.
    mask -- discret mask
    threshold -- see cv2.HoughLinesP
    min_line_length -- see cv2.HoughLinesP
    max_line_gap -- see cv2.HoughLinesP
    return -- numpy array containing hough line count per pixel
    """
    gray = (mask*255).astype('uint8')
    lines = cv2.HoughLinesP(gray,1,np.pi/180,threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)

    hough_lines = np.zeros(gray.shape)
    if not lines is None:
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                one_hough = np.zeros(gray.shape[0:2])
                cv2.line(one_hough,(x1,y1),(x2,y2), 1., 1)
                hough_lines = hough_lines + one_hough

    return hough_lines

def hough_update_mask(mask, hough_lines, kernel, thresh=1, eps=0.2):
    """Updates the mask by increasing probabilities using hough_lines.
    mask -- continous mask, not getting modified
    hough_lines -- result obtained from  get_hough_lines
    kernel -- for morphological closing applied on hough_lines for smoothing them
    If kernel is not None, morphological closing is not applied
    good kernel example np.ones((3,3),np.uint8)
    thresh -- How many lines need to pass through a pixel at least
    eps -- Which constant factor should be added to chosen pixels
    return -- new updated continous mask
    """
    updated_mask = mask.copy()
    hough_lines_c = hough_lines.copy()
    if not kernel is None:
        hough_lines_c = cv2.morphologyEx(hough_lines_c, cv2.MORPH_CLOSE, kernel)
    updated_mask = updated_mask + eps * (hough_lines_c >= thresh)
    return updated_mask

def hough_pipeline(mask, kernel, hough_thresh=100, min_line_length=1,
                    max_line_gap=500, pixel_up_thresh=1, eps=0.2):
    """This method performs the complete update of probability
    maps using the hough transform to detect road segments
    with lower probability.
    mask -- continous mask, not getting modified
    kernel -- for morphological closing applied on hough_lines for smoothing them
    If kernel is not None, morphological closing is not applied
    good kernel example np.ones((3,3),np.uint8)
    hough_thresh -- see cv2.HoughLinesP
    min_line_length -- see cv2.HoughLinesP
    max_line_gap -- see cv2.HoughLinesP
    thresh -- How many lines need to pass through a pixel at least
    eps -- Which constant factor should be added to chosen pixels
    """

    disc_mask = discretize(mask)
    hough_lines = get_hough_lines(mask, threshold=hough_thresh, min_line_length=min_line_length,
                    max_line_gap=max_line_gap)
    updated_mask = hough_update_mask(mask, hough_lines, kernel, thresh=pixel_up_thresh, eps=eps)
    return updated_mask
