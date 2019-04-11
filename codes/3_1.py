
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


# In[ ]:

X = []
Y = {'Y_color':[], 'Y_width':[], 'Y_angle':[], 'Y_length':[]}


# In[ ]:

path = "../2/line_dataset/"
images = os.listdir(path)
random.seed(42)
random.shuffle(images)


# In[ ]:

def Append(image):
    data=image[:-4].split('_')
    length = data[0]
    width = data[1] 
    angle = data[2]
    color = data[3]
    img=cv2.imread(path+image)
    X.append(img)
    Y['Y_color'].append(color)
    Y['Y_length'].append(length)
    Y['Y_width'].append(width)
    Y['Y_angle'].append(angle)


# In[ ]:

for image in images[:10]:
    Append(image)


# In[ ]:

X = np.asarray(X)


# In[ ]:

input_layer = layers.Input(shape=(28,28,3))
x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
# x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
features = Flatten()(x)
x = Dense(64,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output_1 = Dense(1,activation = 'sigmoid',name="color")(x)
x = Dense(64,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output_2 = Dense(1,activation = 'sigmoid',name="length")(x)
x = Dense(64,use_bias=False)(features)
x = BatchNormalization()(x)
x = Activation("relu")(x)
output_3 = Dense(1,activation = 'sigmoid',name="width")(x)
x = Dense(128,use_bias=False)(features)
x = BatchNormalization()(x)
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

# network.summary()
# ann_viz(network,view=True,filename="test.gv",title="network")


# In[ ]:

loss_1 = [binary_crossentropy, binary_crossentropy, binary_crossentropy, categorical_crossentropy]
loss_2 = {"dense_2":"binary_crossentropy","dense_4":"binary_crossentropy","dense_6":"binary_crossentropy","name1":"categorical_crossentropy"}
network.compile(optimizer = 'RMSprop', loss = loss_1, metrics = ['accuracy'])


# In[ ]:

network.load_weights("../2/weights_final.h5")


# In[ ]:

w=28
h=28
fig=plt.figure(figsize=(28, 10))
columns = 5
rows = 2
for i in range(0, columns*rows):
    img = X[i]
    fig.add_subplot(rows, columns, i+1)
    plt.title(i+1)
    plt.imshow(img)
plt.show()

# print("Choose an image from the above image titles:")
img = input("Choose an image from the above image titles:")
img = int(img)-1


# In[ ]:

layer_names=[]
for i,layer in enumerate(network.layers[1:12]):
    layer_names.append(layer.name)
    print(str(i+1)+" "+layer.name)


# In[ ]:

layer_name = input("Select the layer name from above: ")


# In[ ]:

output = network.layers[int(layer_name)].output


# In[ ]:

vis_model = Model(network.input , output)


# In[ ]:

vis_pred=vis_model.predict(X[img:img+1])
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
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image
plt.imsave('1_'+layer_names[int(layer_name)-1]+'.png',display_grid, cmap='viridis')


# In[ ]:



