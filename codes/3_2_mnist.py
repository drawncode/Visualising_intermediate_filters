
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

from keras.datasets import mnist
(X_train,Y_train),(X1_test,Y1_test)=mnist.load_data()

X_train  = X_train.reshape(60000,28,28,1)
from keras.utils import to_categorical
Y_train = np.asarray(Y_train)
Y_train = to_categorical(Y_train)
X_train = np.asarray(X_train)
X_train=X_train.astype('float32')/255.0


split = train_test_split(X_train,Y_train,test_size=0.4, random_state=42)
(X_train,X_test,Y_train,Y_test) = split

print("Data split \n train set :",len(X_train),"\n test set :", len(X_test))


# In[ ]:

input_layer = layers.Input(shape=(28,28,1))
x = Conv2D(32,(3,3),padding = 'same')(input_layer)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32,(3,3),padding = 'same')(x)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32,(3,3),padding = 'same')(x)
x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.5)(x)
features = Flatten()(x)
x = Dense(1024)(features)
x = Activation("relu")(x)
output = Dense(10,activation = 'softmax')(x)
network = Model(input_layer,output)
network.summary()


# In[ ]:

# network.summary()
# ann_viz(network,view=True,filename="test.gv",title="network")


# In[ ]:

loss = categorical_crossentropy
network.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])


# In[ ]:

network.load_weights("weights_mnist_2.h5")

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

def vis_filter(output,filter_index,model):
#     layer_output=model.get_layer(layer_name).output
    loss=k.mean(output[:,:,:,filter_index])
    
    grads=k.gradients(loss,model.layers[0].input)[0]
    
    grads = grads/(k.sqrt(k.mean(k.square(grads))) + 1e-5)
    
    iterate = k.function([model.layers[0].input] , [loss,grads])
    
    img = np.random.random((1, 28, 28, 1)) *20 +128
    
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
display_grid = np.zeros((size * n_cols+n_cols*5, images_per_row * size+images_per_row*5,1))
for col in range(n_cols): # Tiles each filter into a big horizontal grid
    for row in range(images_per_row):
        channel_image = img[col * images_per_row + row]
#         plt.imshow(channel_image)
        display_grid[col * size+col*5 : (col + 1) * size+(col)*5,row * size+row*5 : (row + 1) * size+row*5,:] = channel_image
cv2.imwrite('2_mnist_results/2_mnist'+layer_names[int(layer_name)-1]+'.png',display_grid)

