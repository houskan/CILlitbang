from keras_segmentation.models.unet import resnet50_unet
import cv2
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

model = resnet50_unet(n_classes=2 ,  input_height=416, input_width=416)
"""

for i in range(100):
    path = "training/training/groundtruth/satImage_"+str(i+1).zfill(3)+".png"
    path_save = "training/training/groundtruth_binary/satImage_"+str(i+1).zfill(3)+".bmp"
    img = load_img(path, color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / np.max(img_array)
    img_array[img_array > 0.5] = 1
    img_array[img_array <= 0.5] = 0
    cv2.imwrite( path_save ,img_array )
"""
model.train(
    train_images =  "training/training/images/",
    train_annotations = "training/training/groundtruth_binary/",
    checkpoints_path = "/tmp/resnet50_unet_1" , epochs=1
)

out = model.predict_segmentation(
    inp="training/training/validation_images/satImage_001.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)
"""
# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="training/training/validation_images/"  , annotations_dir="training/training/validation_groundtruth/" ) )
"""
