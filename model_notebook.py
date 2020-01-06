#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[ ]:


train = pd.read_csv('Movies_Poster/Multi_Label_dataset/train.csv')
train.shape[0]


# In[ ]:


# train_image = []
# for i in tqdm(range(train.shape[0])):
#     img = image.load_img('Movies_Poster/Multi_Label_dataset/Images/'+train['Id'][i]+'.jpg',target_size=(400,400,3))
#     img = image.img_to_array(img)
#     img = img/255
#     train_image.append(img)
# X = np.array(train_image)


# In[ ]:


y = np.array(train.drop(['Id', 'Genre'],axis=1))
y.shape


# In[ ]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


X_train = []
X_test = []
y_train = []
y_test = []
def imageLoader(batch_size):
    global X_train
    global X_test
    global y_train
    global y_test
    L = 7254
    train_image = []
    #this line is just to make the generator infinite, keras needs that    
    
    batch_start = 0
    batch_end = batch_size
    while batch_start < L:
        print(batch_start)
        limit = min(batch_end, L)

        train_slice = train.iloc[batch_start:limit]
        # print(train_slice.shape)
        for i in range(train_slice.shape[0]):
            img = image.load_img('Movies_Poster/Multi_Label_dataset/Images/' + train['Id'][i] + '.jpg', target_size=(400,400,3))
            img = image.img_to_array(img)
            print(img)
            # print(img.shape)
            img = img / 255
            train_image.append(img)

        # print(len(train_image))
        
        X = np.array(train_image[batch_start:limit])
        Y = y[batch_start:limit]


        # X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.1)
        print(X.shape)
        print(Y.shape)

        # yield (X_train, y_train)
        yield (X, Y)

        batch_start += batch_size   
        batch_end += batch_size

            

images = imageLoader(500)
# image = next(images)
# print(next(images))
# for index, i in enumerate(images):
#     print(index)
    # print(len(i[0]), len(i[1]))
model.fit_generator(images, steps_per_epoch=14, epochs=10, validation_data=(X_test, y_test), verbose=2, initial_epoch=1)
# model.fit(image, steps_per_epoch=14, epochs=10, validation_data=(X_test, y_test), verbose=2, initial_epoch=1)




