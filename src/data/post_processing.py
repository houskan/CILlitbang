import numpy as np
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte, measure
import cv2

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
    hough_lines = get_hough_lines(disc_mask, threshold=hough_thresh, min_line_length=min_line_length,
                    max_line_gap=max_line_gap)
    updated_mask = hough_update_mask(mask, hough_lines, kernel, thresh=pixel_up_thresh, eps=eps)
    return updated_mask


def predict_combined(model, target_size, file_name, img_folder, mask_folder, combined_folder=None, mode='avg'):

    # Getting image name (without extension)
    img_name = file_name.split('.')[-2]

    # Reading image and converting it to float range [0, 1]
    img = io.imread(os.path.join(img_folder, file_name), as_gray=False)
    img = img / 255.0

    # Saving original size and resizing image to target size of model
    original_size = img.shape
    img = trans.resize(img, target_size)

    # Constructing all eight rotated and flipped input images
    img0 = img.copy()
    img1 = np.rot90(img, 1)
    img2 = np.rot90(img, 2)
    img3 = np.rot90(img, 3)
    img4 = np.fliplr(img)
    img5 = np.rot90(np.fliplr(img), 1)
    img6 = np.rot90(np.fliplr(img), 2)
    img7 = np.rot90(np.fliplr(img), 3)

    # Predicting results for eight rotated and flipped input images
    results = model.predict(np.array([img0, img1, img2, img3, img4, img5, img6, img7]), batch_size=8, verbose=1)

    # Reversing rotation and flipping (mirroring) of resulting continuous output masks
    result0 = results[0]
    result1 = np.rot90(results[1], 3)
    result2 = np.rot90(results[2], 2)
    result3 = np.rot90(results[3], 1)
    result4 = np.fliplr(results[4])
    result5 = np.fliplr(np.rot90(results[5], 3))
    result6 = np.fliplr(np.rot90(results[6], 2))
    result7 = np.fliplr(np.rot90(results[7], 1))

    # Resizing resulting output masks to original size
    result0 = trans.resize(result0, original_size)
    result1 = trans.resize(result1, original_size)
    result2 = trans.resize(result2, original_size)
    result3 = trans.resize(result3, original_size)
    result4 = trans.resize(result4, original_size)
    result5 = trans.resize(result5, original_size)
    result6 = trans.resize(result6, original_size)
    result7 = trans.resize(result7, original_size)

    # Converting continuous output results to discrete masks with zero and one values
    result0_disc = discretize(result0)
    result1_disc = discretize(result1)
    result2_disc = discretize(result2)
    result3_disc = discretize(result3)
    result4_disc = discretize(result4)
    result5_disc = discretize(result5)
    result6_disc = discretize(result6)
    result7_disc = discretize(result7)

    # Checking if combined folder valid and saving intermediate continuous output masks
    if not (combined_folder is None):
        io.imsave(os.path.join(combined_folder, img_name + '_comb0_cont.png'), img_as_ubyte(result0))
        io.imsave(os.path.join(combined_folder, img_name + '_comb1_cont.png'), img_as_ubyte(result1))
        io.imsave(os.path.join(combined_folder, img_name + '_comb2_cont.png'), img_as_ubyte(result2))
        io.imsave(os.path.join(combined_folder, img_name + '_comb3_cont.png'), img_as_ubyte(result3))
        io.imsave(os.path.join(combined_folder, img_name + '_comb4_cont.png'), img_as_ubyte(result4))
        io.imsave(os.path.join(combined_folder, img_name + '_comb5_cont.png'), img_as_ubyte(result5))
        io.imsave(os.path.join(combined_folder, img_name + '_comb6_cont.png'), img_as_ubyte(result6))
        io.imsave(os.path.join(combined_folder, img_name + '_comb7_cont.png'), img_as_ubyte(result7))

    # Checking if combined folder valid and saving intermediate discrete output masks
    if not (combined_folder is None):
        io.imsave(os.path.join(combined_folder, img_name + '_comb0.png'), img_as_ubyte(result0_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb1.png'), img_as_ubyte(result1_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb2.png'), img_as_ubyte(result2_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb3.png'), img_as_ubyte(result3_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb4.png'), img_as_ubyte(result4_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb5.png'), img_as_ubyte(result5_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb6.png'), img_as_ubyte(result6_disc))
        io.imsave(os.path.join(combined_folder, img_name + '_comb7.png'), img_as_ubyte(result7_disc))

    # Checking which mode of combining should be used
    if mode == 'avg':
        result_avg = (1.0 / 8.0) * (result0 + result1 + result2 + result3 + result4 + result5 + result6 + result7)
        result_avg = trans.resize(result_avg, original_size)
        io.imsave(os.path.join(mask_folder, img_name + '_cont.png'), img_as_ubyte(result_avg))
        result_avg_disc = discretize(result_avg)
        io.imsave(os.path.join(mask_folder, img_name + '.png'), img_as_ubyte(result_avg_disc))
        return result_avg_disc
    elif mode == 'vote':
        thresh = 5
        result_vote_sum = (result0_disc + result1_disc + result2_disc + result3_disc + result4_disc + result5_disc + result6_disc + result7_disc)
        result_vote_avg = (1.0 / 8.0) * result_vote_sum
        io.imsave(os.path.join(mask_folder, img_name + '_cont.png'), img_as_ubyte(result_vote_avg))
        result_vote_disc = result_vote_sum.copy()
        result_vote_disc[result_vote_disc < thresh] = 0.0
        result_vote_disc[result_vote_disc >= thresh] = 1.0
        io.imsave(os.path.join(mask_folder, img_name + '.png'), img_as_ubyte(result_vote_disc))
        return result_vote_disc

    return None


def discretize(result_cont):
    result_disc = result_cont.copy()
    result_disc[result_disc > 0.5] = 1.0
    result_disc[result_disc <= 0.5] = 0.0
    return result_disc

def saveCombinedResult(model, test_path, image_folder):

    folder = os.path.join(test_path, image_folder)
    for file in os.listdir(folder):
        predict_combined(model=model, target_size=(400, 400),
                         file_name=file,
                         img_folder=folder,
                         mask_folder=os.path.join(test_path, 'results'),
                         combined_folder=os.path.join(test_path, 'results', 'combined'),
                         #combined_folder=None,
                         mode='avg')

def saveResult(test_path, images, results):

    # Initializing two list for test image mask result file path names (first discrete, second continuous)
    resultNames = list(map(lambda x: os.path.join(test_path, 'results', x), images))
    resultNamesCont = list(map(lambda x: os.path.join(test_path, 'results', '{0}_cont.{1}'.format(*x.rsplit('.', 1))), images))

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
