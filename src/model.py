#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import json
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import keras.metrics
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil

# In[ ]:

train = pd.read_csv('src/Movies_Poster/Multi_Label_dataset/train.csv')
train.shape[0]


# In[ ]:
# y = np.array(train.drop(['Id', 'Genre'],axis=1))
# y.shape
# X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=42, test_size=0.1)

# path = os.getcwd()
# print(path)

# x_test = X_test.drop(['Genre'], axis=1)

# export_csv = x_test.to_csv(os.path.join(path, 'test.csv'), index=False)

# imgs = []
# for name in X_test['Id']:
    # img = image.load_img('src/Movies_Poster/Multi_Label_dataset/Images/' + name + '.jpg', target_size=(400,400,3))
    # newpath = shutil.copy('src/Movies_Poster/Multi_Label_dataset/Images/' + name + '.jpg', 'src/Movies_Poster/Multi_Label_dataset/Images/test/' + name + '.jpg', )
    # print(newpath)
    # img = image.img_to_array(img)
    # print(img.shape)
#     img = img / 255
#     imgs.append(img)
# X_test = np.array(imgs)


# In[ ]:

#Base Line
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='sigmoid'))


# Modelo 2
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(400,400,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='sigmoid'))

# Modelo 3

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(400, 400, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='sigmoid'))


# In[ ]:
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

L = len(X_train)

# In[ ]:
# class BatchSequence(keras.utils.Sequence):
#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.x) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         train_image = []
#         for name in self.x.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]['Id']:
#             img = image.load_img('src/Movies_Poster/Multi_Label_dataset/Images/' + name + '.jpg', target_size=(400,400,3))
#             img = image.img_to_array(img)
#             img = img / 255
#             train_image.append(img)
        
#         batch_x = np.array(train_image)
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

#         return batch_x, batch_y


# In[]:
# batch_size = 50
# L = len(X_train)
# train_batch = BatchSequence(X_train, y_train, batch_size)
# model.fit_generator(train_batch, epochs=8, validation_data=(X_test, y_test))


# In[]:

# Model Metrics
import json
from utils import metrics
from keras.models import load_model

dict_ = {
    'BaseLine':  metrics(load_model('dist/model2.h5'), (400,400), 'src/Movies_Poster/Multi_Label_dataset/Images/test/','test.csv'),
    'Model2': metrics(load_model('dist/model3.h5'), (400,400), 'src/Movies_Poster/Multi_Label_dataset/Images/test/','test.csv'),
    'Model3': metrics(load_model('dist/model4.h5'), (400,400), 'src/Movies_Poster/Multi_Label_dataset/Images/test/','test.csv'),
}

with open('metrics.json', 'w') as of:
    json.dump(dict_, of)


# Save Model
# path = os.getcwd()

# model.save(os.path.join(path, "dist", "model4.h5"))
# print(f"Saved model to disk")



# Predict Model
# model = load_model("dist/model4.h5")

# model.summary()

# img = image.load_img('src/Movies_Poster/Movie_Poster_Dataset/2015/tt2293640.jpg',target_size=(400,400,3))
# img = image.img_to_array(img)
# img = img/255

# classes = np.array(train.columns[2:])
# proba = model.predict(img.reshape(1,400,400,3))
# top_3 = np.argsort(proba[0])[:-10:-1]
# for i in range(9):
#     print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
# plt.imshow(img)



# %%
