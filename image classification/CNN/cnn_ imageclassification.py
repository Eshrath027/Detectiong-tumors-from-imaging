#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
import keras.optimizers
from sklearn.metrics import classification_report
import keras.optimizers
from keras.applications import vgg16
import numpy as np
import random
import os
from tqdm import tqdm
import pickle
import cv2


# In[2]:



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import pickle
import time
import numpy as np
import keras.optimizers
from sklearn.metrics import classification_report


# In[19]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')


# In[3]:


# Define necessary constants
TEST_DIR = '/content/drive/MyDrive/archive (5)/Testing'
TRAIN_DIR = '/content/drive/MyDrive/archive (5)/Training'
IMG_SIZE = 224
CATEGORIES = ["glioma","meningioma","notumor","pituitary"]


# In[5]:


# Creating training dataset
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DIR,category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
          img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          training_data.append([new_array, class_num])

    random.shuffle(training_data)

create_training_data()
#np.save('train_data.npy', training_data)
print(len(training_data))

print("train")
print()
X_train = np.array([i[0] for i in training_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y_train = [i[1] for i in training_data]

pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("Y_train.pickle","wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()


# In[4]:


# Creating testing dataset
testing_data = []

def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(TEST_DIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
          img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          testing_data.append([new_array, class_num])

    random.shuffle(testing_data)

create_testing_data()
#np.save('testing_data.npy', testing_data)
print(len(testing_data))

print("testing")
print()
X_test= np.array([i[0] for i in testing_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y_test = [i[1] for i in testing_data]

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("Y_test.pickle","wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()


# In[6]:


#print(X_train.shape)
print(X_test.shape)


# In[8]:


X_train=X_train[0:2000]
Y_train=Y_train[0:2000]


# In[5]:


#X_train = X_train / 255.0
X_test = X_test / 255.0

#Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# In[10]:


from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
import keras.optimizers
from sklearn.metrics import classification_report
import keras.optimizers

import numpy as np


# In[11]:


dense_layers = [0,1,2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
            if NAME == "3-conv-128-nodes-1-dense":
              print(NAME)


# In[12]:



tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


# In[13]:


dense_layers = [0,1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
            if NAME == "3-conv-128-nodes-1-dense":
              print(NAME)
              model = Sequential()

              model.add(Conv2D(layer_size, (3, 3), input_shape=X_train.shape[1:]))
              model.add(Activation('relu'))
              model.add(MaxPooling2D(pool_size=(2, 2)))

              for l in range(conv_layer-1):
                  model.add(Conv2D(layer_size, (3, 3)))
                  model.add(Activation('relu'))
                  model.add(MaxPooling2D(pool_size=(2, 2)))

              model.add(Flatten())
              for _ in range(dense_layer):
                  model.add(Dense(layer_size))
                  model.add(Activation('relu'))
                  model.add(Dropout(0.33))

              model.add(Dense(4))
              model.add(Activation('softmax'))


              model.compile(loss='sparse_categorical_crossentropy',
                optimizer= "adam",
                metrics=['accuracy'],
                )

              #Fit the model
              model.fit(X_train, Y_train,
              batch_size=32,
              epochs=20,
              validation_data=(X_test,Y_test),
              callbacks=[tensorboard,es])


              #Save model
              model.save("{}-model.h5".format(NAME))


# In[9]:


from keras.models import load_model
model = load_model('/content/3-conv-128-nodes-1-dense-model.h5',compile=True)


# In[10]:


scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[8]:


y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(Y_test, y_pred_bool))


# In[ ]:




