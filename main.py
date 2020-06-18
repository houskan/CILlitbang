import numpy as np
import tensorflow as tf


tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)

from model import *
from data import *

data_gen_args = dict(rotation_range=0.5,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

trainGen = trainGenerator(2, 'training/training', 'images', 'groundtruth', data_gen_args)

model = unet()

# To save the model
#model_checkpoint = ModelCheckpoint('unet_roadseg.hdf5', monitor='loss', verbose=1, save_best_only=True)
#model.fit(trainGen, steps_per_epoch=300, epochs=3, callbacks=[model_checkpoint], verbose=1)

model.fit(trainGen, steps_per_epoch=100, epochs=10, verbose=1)

testGen = testGenerator("test_images/test_images", num_image=90)
results = model.predict(testGen, steps=30, verbose=1)
saveResult("test_images/results", results)

