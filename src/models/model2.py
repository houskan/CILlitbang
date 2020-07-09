from tensorflow.keras.layers import *
from tensorflow.keras import Model

'''
Helper function for the Implementation of the
VGG-Net Variant in https://arxiv.org/pdf/1607.05620.pdf.
Implements up to the Dense Layer that reduces the
dimensions.
arg: Input Tensor (Input(...)), height and width of output
return: output of network
'''
def _vgg_net(inp, output_height, output_width):
    o = Conv2D(64, (3,3), padding='same', activation='relu')(inp)
    o = Conv2D(64, (3,3), padding='same', activation='relu')(o)
    o = MaxPooling2D((2,2), strides=(2,2))(o)

    o = Conv2D(128, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(128, (3,3), padding='same', activation='relu')(o)
    o = MaxPooling2D((2,2), strides=(2,2))(o)

    o = Conv2D(256, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(256, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(256, (3,3), padding='same', activation='relu')(o)
    o = MaxPooling2D((2,2), strides=(2,2))(o)

    o = Conv2D(512, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3,3), padding='same', activation='relu')(o)
    o = MaxPooling2D((2,2), strides=(2,2))(o)
    
    o = Conv2D(512, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3,3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3,3), padding='same', activation='relu')(o)
    o = MaxPooling2D((2,2), strides=(2,2))(o)

    o = Flatten()(o)
    o = Dense(4096, activation='relu')(o)
    o = Dropout(0.5)(o)
    o = Dense(4096, activation='relu')(o)

    return o

'''
Helper function for the Implementation of the
AlexNet Variant in https://arxiv.org/pdf/1607.05620.pdf.
Implements up to the Dense Layer that reduces the
dimensions.
arg: Input Tensor (Input(...)), height and width of output
return: output of network
'''
def _alex_net(inp, output_height=16, output_width=16):
    o = Conv2D(96, (11,11), strides=(4,4), padding='same', activation='relu')(inp)
    o = MaxPooling2D((3,3), strides=(2,2))(o)

    o = Conv2D(64, (5,5), padding='same', activation='relu')(o)
    o = MaxPooling2D((3,3), strides=(2,2))(o)

    o = Conv2D(384, (3,3), padding='same', activation='relu')(o)

    o = Conv2D(384, (3,3), padding='same', activation='relu')(o)

    o = Conv2D(256, (3,3), padding='same', activation='relu')(o)
    o = MaxPooling2D((3,3), strides=(2,2))(o)

    o = Flatten()(o)

    o = Dense(4096, activation='relu')(o)
    o = Dropout(0.5)(o)
    o = Dense(4096, activation='relu')(o)

    return o

'''
Function returning the VGG-Net Variant in https://arxiv.org/pdf/1607.05620.pdf.
arg: input/output heights 
return: Model
'''
def vgg_net_model(input_height=64, input_width=64, output_height=16, output_width=16):
   inp = Input(shape=(input_height, input_width, 3)) 
   o = _vgg_net(inp, output_height, output_width)
   o = Dense(output_height*output_width, activation='relu')(o)
   o = Reshape((output_height, output_width))(o)
   model = Model(inp, o)
   return model

'''
Function returning the Alex-Net Variant in https://arxiv.org/pdf/1607.05620.pdf.
arg: input/output heights 
return: Model
'''
def alex_net_model(input_height=256, input_width=256, output_height=16, output_width=16):
   inp = Input(shape=(input_height, input_width, 3)) 
   o = _alex_net(inp, output_height, output_width)
   o = Dense(output_height*output_width, activation='relu')(o)
   o = Reshape((output_height, output_width))(o)
   model = Model(inp, o)
   return model

'''
Function returning the LG-Seg-Net in https://arxiv.org/pdf/1607.05620.pdf.
arg: input/output heights 
return: Model
'''
def lg_seg_model(input_height_vgg=256, input_width_vgg=256, input_height_alex=256, input_width_alex=256, output_height=16, output_width=16):
   inp_vgg = Input(shape=(input_height_vgg, input_width_vgg, 3)) 
   inp_alex = Input(shape=(input_height_alex, input_width_alex, 3)) 

   o1 = _vgg_net(inp_vgg, output_height, output_width)
   o1 = Flatten()(o1)

   o2 = _alex_net(inp_alex, output_height, output_width)
   o2 = Flatten()(o2)

   o = Concatenate()([o1,o2])

   o = Dense(4096, activation='relu')(o)
   o = Dense(256, activation='relu')(o)
   o = Dense(output_height*output_width, activation='relu')(o)
   o = Reshape((output_height, output_width))(o)
   model =  Model([inp_vgg, inp_alex], o)
   return model
