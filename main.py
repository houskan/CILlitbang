import numpy as np
import tensorflow as tf 

from model import *
from data import *

test_path = 'test_images'

train_batch = trainGenerator()
validation_batch = validationGenerator()

model = unet()
model.fit_generator(train_batch, steps_per_epoch=100, validation_data=validation_batch, validation_steps=2, epochs=500, verbose=1)

test_batch = testGenerator()
results = model.predict_generator(test_batch, 2, verbose=1)
results = results
saveResult("test_images/results", results)
