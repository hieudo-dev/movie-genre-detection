#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


# In[ ]:

path_prefix = os.path.join(path, "src", "Movies_Poster", "Multi_Label_dataset")
train = pd.read_csv(os.join(path_prefix, "train.csv"))
train.shape[0]


# In[ ]:


y = np.array(train.drop(['Id', 'Genre'],axis=1))
y.shape
X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=42, test_size=0.1)

imgs = []
for name in X_test['Id']:
    img = image.load_img(os.join(path_prefix, "Images", name, ".jpg"), target_size=(400,400,3))
    img = image.img_to_array(img)
    # print(img.shape)
    img = img / 255
    imgs.append(img)
X_test = np.array(imgs)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=5, strides=2, activation='relu', input_shape=(400,400,3)))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))       

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(25, activation='sigmoid'))   # Final Layer using Softmax



# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

L = len(X_train)

# In[ ]:
class BatchSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        train_image = []
        for name in self.x.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]['Id']:
            img = image.load_img(os.join(path_prefix, "Images", name, ".jpg"), target_size=(400,400,3))
            img = image.img_to_array(img)
            img = img / 255
            train_image.append(img)
        
        batch_x = np.array(train_image)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


# In[]:
batch_size = 50
L = len(X_train)
batch = BatchSequence(X_train, y_train, batch_size)
model.fit_generator(batch, epochs=5, validation_data=(X_test, y_test))


path = os.getcwd() # Current Path

model.save(os.path.join(path, "dist", "model.h5"))
print(f"Saved model to disk")
