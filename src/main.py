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

trainGen = trainResnetGenerator(2, train_path, 'images', 'groundtruth', 2, input_height, input_width, output_height, output_width)
testGen = testResnetGenerator(test_path, 'images', input_height, input_width)

model.fit(trainGen, steps_per_epoch=100, epochs=5, verbose=1)

images = os.listdir(os.path.join(test_path, 'images'))
results = model.predict(testGen, steps=len(images), verbose=1)
saveResnetResult(test_path, images, results, output_height, output_width, n_classes)

