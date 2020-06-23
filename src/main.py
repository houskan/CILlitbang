import numpy as np
import tensorflow as tf

from models.model import *
from data.data import *

train_path = '../data/training/'
test_path = '../data/test/'
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest',
                     validation_split=0.2)

trainGen, validationGenerator = getTrainGenerators(data_gen_args, train_path, test_path, batch_size=2)

model = unet()

# To save the model
# model_checkpoint = ModelCheckpoint('unet_roadseg.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit(trainGen, steps_per_epoch=300, epochs=3, callbacks=[model_checkpoint], verbose=1)

model.fit(trainGen, validation_data=validationGenerator, steps_per_epoch=100, validation_steps=10, epochs=10, verbose=1)

images = os.listdir(os.path.join(test_path, 'images'))
testGen = testGenerator(test_path, num_image=len(images))
results = model.predict(testGen, steps=len(images), verbose=1)
saveResult(test_path, results)
