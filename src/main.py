import numpy as np
import tensorflow as tf

from models.model import *
from data.data import *

train_path = '../data/training/'
test_path = '../data/test/'

model = resnet50_unet(n_classes=2, input_height=608, input_width=608)

n_classes = model.n_classes
input_height = model.input_height
input_width = model.input_width
output_height = model.output_height
output_width = model.output_width

trainGen, valGen = getResnetGenerators(train_path=train_path, image_folder='images', mask_folder='groundtruth', 
                                       input_height=input_height, input_width=input_width, output_height=output_height, output_width=output_width, 
                                       n_classes=2, batch_size=2, validation_split=0.1)

testGen = testResnetGenerator(test_path=test_path, image_folder='images', input_height=input_height, input_width=input_width)

model.fit(trainGen, steps_per_epoch=100, epochs=5, validation_data=valGen, validation_steps=10, verbose=1)

images = os.listdir(os.path.join(test_path, 'images'))
results = model.predict(testGen, steps=len(images), verbose=1)
saveResnetResult(test_path=test_path, images=images, results=results, output_height=output_height, output_width=output_width, n_classes=n_classes)

