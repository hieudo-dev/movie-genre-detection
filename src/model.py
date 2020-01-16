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
X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=42, test_size=0.1)

imgs = []
for name in X_test['Id']:
    img = image.load_img('Movies_Poster/Multi_Label_dataset/Images/' + name + '.jpg', target_size=(400,400,3))
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


# In[ ]:
def imageLoader(batch_size):
    L = len(X_train)
    
    batch_start = 0
    batch_end = batch_size
    while batch_start < L:
        print(batch_start)
        train_image = []
        limit = min(batch_end, L)

        # print(train_slice.shape)
        for name in X_train.iloc[batch_start:limit]['Id']:
            img = image.load_img('Movies_Poster/Multi_Label_dataset/Images/' + name + '.jpg', target_size=(400,400,3))
            img = image.img_to_array(img)
            # print(img.shape)
            img = img / 255
            train_image.append(img)

        # print(len(train_image))
        
        batch_X = np.array(train_image)
        batch_Y = y_train[batch_start:limit]

        yield (batch_X, batch_Y)

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





