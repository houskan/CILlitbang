import numpy as np
import tensorflow as tf

from models.model import *
from data.data import *

from skimage import img_as_ubyte

colors = [(0,0,0), (255, 255, 255)]


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

#trainGen, validationGenerator = getTrainGenerators(data_gen_args, train_path, test_path, batch_size=2)

model = resnet50_unet(n_classes=2, input_height=608, input_width=608)

n_classes = model.n_classes
input_height = model.input_height
input_width = model.input_width
output_height = model.output_height
output_width = model.output_width

trainGen = trainResnetGenerator(2, train_path, 'images', 'groundtruth', 2, input_height, input_width, output_height, output_width)

model.fit(trainGen, steps_per_epoch=100, epochs=5, verbose=1)
# To save the model
# model_checkpoint = ModelCheckpoint('unet_roadseg.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit(trainGen, steps_per_epoch=300, epochs=3, callbacks=[model_checkpoint], verbose=1)

#model.fit(trainGen, validation_data=validationGenerator, steps_per_epoch=100, validation_steps=10, epochs=10, verbose=1)

#testGen = testGenerator(test_path, num_image=90)
testGen = testResnetGenerator(test_path, 'images', input_height, input_width)
#results = model.predict(testGen, steps=94, verbose=1)
#saveResult(os.path.join(test_path, 'results'), results, output_height, output_width, n_classes)

"""
Will adapt this code into saveResult as soon as I can test stuff again on my desktop PC - Steven
"""
test_img = cv2.imread(os.path.join(test_path, 'images/test_7.png'))
test_img = cv2.resize(test_img, (input_width, input_height))
test_img = test_img.astype(np.float32)
test_img[:, :, 0] -= 103.939
test_img[:, :, 1] -= 116.779
test_img[:, :, 2] -= 123.68
test_img = test_img[:, :, ::-1]
result = model.predict(np.array([test_img]))[0]
result = result.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
seg_img = np.zeros((output_height, output_width, 3))
for c in range(n_classes):
	seg_arr_c = result[:, :] == c
	seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
	seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
	seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

img = cv2.resize(seg_img, (400, 400))
cv2.imwrite(os.path.join(test_path, "resnet_predict.png"), img)

#images = os.listdir(os.path.join(test_path, 'images'))
#testGen = testGenerator(test_path, num_image=len(images))
#results = model.predict(testGen, steps=len(images), verbose=1)
#saveResult(test_path, results)
