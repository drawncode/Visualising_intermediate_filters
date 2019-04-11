
# coding: utf-8

# In[ ]:

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.layers import *
from keras import layers
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy, categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import os
import cv2
import sys
import pydot
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras
from keras.layers import deserialize as layer_from_config
from keras import backend as k


# In[ ]:

input_layer = layers.Input(shape=(28,28,3))
x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(input_layer)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
features = Flatten()(x)
x = Dense(64,use_bias=False)(features)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
output_1 = Dense(1,activation = 'sigmoid',name="color")(x)
x = Dense(64,use_bias=False)(features)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
output_2 = Dense(1,activation = 'sigmoid',name="length")(x)
x = Dense(64,use_bias=False)(features)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
output_3 = Dense(1,activation = 'sigmoid',name="width")(x)
x = Dense(128,use_bias=False)(features)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Dense(256,use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
# x = Dense(128,use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
output_4 = Dense(12,activation = 'softmax',name="angle")(x)
network = Model(input_layer,[output_1,output_2,output_3,output_4])
# plot_model(network, to_file='model.png')


# In[ ]:

def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.
    # Arguments
        x: A numpy-array representing the generated image.
    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[ ]:

network.load_weights("../2/no_batch.h5")


# In[ ]:

def vis_filter(output,filter_index,model):
#     layer_output=model.get_layer(layer_name).output
    loss=k.mean(output[:,:,:,filter_index])
    
    grads=k.gradients(loss,model.layers[0].input)[0]
    
    grads = grads/(k.sqrt(k.mean(k.square(grads))) + 1e-5)
    
    iterate = k.function([model.layers[0].input] , [loss,grads])
    
    img = np.random.random((1, 28, 28, 3)) *20 +128
    
    for i in range(100):
        loss_value,grads_value = iterate([img])
        img = img + grads_value
#         print("loss value = ",i,loss_value)
#    print(filter_index ,loss_value)
    img = img[0,:,:,:]
    img = deprocess_image(img)
    return img
    


# In[ ]:

layer_names=[]
for i,layer in enumerate(network.layers[1:9]):
    layer_names.append(layer.name)
    print(str(i+1)+" "+layer.name)


# In[ ]:

layer_name = input("Select the layer name from above: ")


# In[ ]:

output = network.layers[int(layer_name)].output
num_filters = int(output.shape[3])
img=[]
print("starting the gradient descent for the filters")
for i in range(num_filters):
    img.append(vis_filter(output,i,network))


# In[ ]:

images_per_row = 16
n_features = len(img) # Number of filters
size = img[0].shape[0] # Width or height of image
n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
display_grid = np.zeros((size * n_cols+n_cols*5, images_per_row * size+images_per_row*5,3))
for col in range(n_cols): # Tiles each filter into a big horizontal grid
    for row in range(images_per_row):
        channel_image = img[col * images_per_row + row]
#         plt.imshow(channel_image)
        display_grid[col * size+col*5 : (col + 1) * size+(col)*5,row * size+row*5 : (row + 1) * size+row*5,:] = channel_image
cv2.imwrite('2_'+layer_names[int(layer_name)-1]+'.png',display_grid)

