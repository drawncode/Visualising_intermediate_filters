
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


X = []
Y = []


path = "../2/line_dataset/"
images = os.listdir(path)
random.seed(42)
random.shuffle(images)

def eval_class(data):
    length = data[0]
    width = data[1]     
    angle = data[2]
    colour = data[3]
    base = 0
    if length=='0':
        if width == '0':
            if colour == '0':
                class_id = int(angle)
            else:
                class_id = 12 + int(angle)
        else:
            if colour == '0':
                class_id = 24+int(angle)
            else:
                class_id = 36+ int(angle)
    else:
        if width == '0':
            if colour == '0':
                class_id = 48+int(angle)
            else:
                class_id = 60 + int(angle)
        else:
            if colour == '0':
                class_id = 72+int(angle)
            else:
                class_id = 84+ int(angle)
    return class_id

print("loading the data...........")
for image in images[:10]:
    img=cv2.imread(path+image)
    X.append(img)
    data=image[:-4].split('_')
    Y.append(eval_class(data))
print(len(X), "images loaded successfully.")

X = np.asarray(X)
X=X.astype('float32')/255.0
Y=np.asarray(Y)
Y=to_categorical(Y,96)
print(X.shape)


# print("Data split \n train set :",len(X_train),"\n test set :", len(X_test))
input_layer = layers.Input(shape=(28,28,3))
x = Conv2D(32,(3,3),padding = 'same')(input_layer)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),padding = 'same')(x)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),padding = 'same')(x)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
features = Flatten()(x)
x = Dense(1024)(features)
x = Activation("relu")(x)
output = Dense(96,activation = 'softmax')(x)
network = Model(input_layer,output)
# network.summary()

loss = categorical_crossentropy
network.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])



# In[ ]:

network.load_weights("weights_line_2.h5")


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
        # channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image
plt.imsave('1_line_results/1_line_'+layer_names[int(layer_name)-1]+'.png',display_grid, cmap='viridis')


# In[ ]:



