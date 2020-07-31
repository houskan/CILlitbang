import cv2
import os
import numpy as np

from keras_seg_submission import *

from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from keras_segmentation.models.unet import resnet50_unet


train_path = "../../../data/training_original/"
val_path = "../../../data/validation_original/"
test_path = "../../../data/test/"

binary_groundtruth_path = "./groundtruth/"
results_path = "./results/"

if not os.path.exists(binary_groundtruth_path):
    os.mkdir(binary_groundtruth_path)
    os.mkdir(os.path.join(binary_groundtruth_path, "train"))
    os.mkdir(os.path.join(binary_groundtruth_path, "validation"))

if not os.path.exists(results_path):
    os.mkdir(results_path)

# Produce binary mask (range [0, 1]) from original groundtruth (range [0, 255])
for f in os.listdir(os.path.join(train_path, "groundtruth")):
    path = os.path.join(train_path, "groundtruth", f)
    path_save = os.path.join(binary_groundtruth_path, "train", f)
    img = load_img(path, color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / np.max(img_array)
    img_array[img_array > 0.5] = 1
    img_array[img_array <= 0.5] = 0
    cv2.imwrite(path_save, img_array)

# Produce binary mask (range [0, 1]) from original groundtruth (range [0, 255])
for f in os.listdir(os.path.join(val_path, "groundtruth")):
    path = os.path.join(val_path, "groundtruth", f)
    path_save = os.path.join(binary_groundtruth_path, "validation", f)
    img = load_img(path, color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / np.max(img_array)
    img_array[img_array > 0.5] = 1
    img_array[img_array <= 0.5] = 0
    cv2.imwrite(path_save, img_array)


model = resnet50_unet(n_classes=2, input_height=416, input_width=416)

model.train(
    train_images=os.path.join(train_path, "images"),
    train_annotations=os.path.join(binary_groundtruth_path, "train"),
    val_images=os.path.join(val_path, "images"),
    val_annotations=os.path.join(binary_groundtruth_path, "validation"),
    do_augment=True,
    validate=True,
    epochs=50,
    steps_per_epoch=100,
    val_steps_per_epoch=10,
    batch_size=4
)

pred_colors = [(0, 0, 0), (255, 255, 255)]

for f in os.listdir(os.path.join(test_path, "images")):
    out = model.predict_segmentation(inp=os.path.join(test_path, "images", f), 
                                     out_fname=os.path.join(results_path, f),
                                     colors=pred_colors)

create_submission()
