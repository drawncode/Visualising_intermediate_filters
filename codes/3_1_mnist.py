
# coding: utf-8

# In[ ]:

import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
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
# network.summary()
# ann_viz(network,view=True,filename="test.gv",title="network")


# In[ ]:

loss = categorical_crossentropy
network.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])


# In[ ]:

network.load_weights("weights_mnist_2.h5")
# print(X_test.shape)

# In[ ]:

w=28
h=28
fig=plt.figure(figsize=(28, 10))
columns = 5
rows = 2
for i in range(0, columns*rows):
    img = X_test[i,:,:,0]
    # print(img.shape)
    fig.add_subplot(rows, columns, i+1)
    plt.title(i+1)
    plt.imshow(img)
plt.show()

# print("Choose an image from the above image titles:")
img = input("Choose an image from the above image titles:")
img = int(img)-1


# In[ ]:

layer_names=[]
for i,layer in enumerate(network.layers[1:9]):
    layer_names.append(layer.name)
    print(str(i+1)+" "+layer.name)


# In[ ]:

layer_name = input("Select the layer name from above: ")


# In[ ]:

output = network.layers[int(layer_name)].output


# In[ ]:

vis_model = Model(network.input , output)


# In[ ]:

vis_pred=vis_model.predict(X_test[img:img+1])
# vis_pred[0].shape


# In[ ]:

images_per_row = 16
layer_activation=vis_pred[0]
n_features = layer_activation.shape[-1] # Number of filters
size = layer_activation.shape[1] # Width or height of image
n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
display_grid = np.zeros((size * n_cols, images_per_row * size))
for col in range(n_cols): # Tiles each filter into a big horizontal grid
    for row in range(images_per_row):
        channel_image = layer_activation[:, :,col * images_per_row + row]
        channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
        # channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image
plt.imsave('1_mnist_results/1_mnist_'+layer_names[int(layer_name)-1]+'.png',display_grid, cmap='viridis')


# In[ ]:



