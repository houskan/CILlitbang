import numpy as np
import cv2
import skimage.io as io
import skimage.transform as trans

import os

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
    result0 = model.predict(np.array([img0]))[0]
    result1 = model.predict(np.array([img1]))[0]
    result2 = model.predict(np.array([img2]))[0]
    result3 = model.predict(np.array([img3]))[0]
    result4 = model.predict(np.array([img4]))[0]
    result5 = model.predict(np.array([img5]))[0]
    result6 = model.predict(np.array([img6]))[0]
    result7 = model.predict(np.array([img7]))[0]

    # Reversing rotation and flipping (mirroring) of resulting continuous output masks
    #result0 = result0
    result1 = np.rot90(result1, 3)
    result2 = np.rot90(result2, 2)
    result3 = np.rot90(result3, 1)
    result4 = np.fliplr(result4)
    result5 = np.fliplr(np.rot90(result5, 3))
    result6 = np.fliplr(np.rot90(result6, 2))
    result7 = np.fliplr(np.rot90(result7, 1))

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
    for i, file in enumerate(os.listdir(folder)):
        predict_combined(model=model, target_size=(400, 400),
                         file_name=file,
                         img_folder=folder,
                         mask_folder=os.path.join(test_path, 'results'),
                         combined_folder=os.path.join(test_path, 'results', 'combined'),
                         #combined_folder=None,
                         mode='avg')
