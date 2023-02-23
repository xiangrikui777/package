# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:35:50 2019

@author: zyp
"""

from keras import layers
from keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

train_dir = ''
validation_dir =''
#r'D:\kaggle\\dogsvscats\\cats_and_dogs_small\\'




train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')





model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


'''
# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
    
'''
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def Specificy(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    TN=K.sum(1 - y_true-y_pred + y_pred * y_true)
    return TN/N
    
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc




model.compile(optimizer=Adam(lr = 1e-4),loss='binary_crossentropy', metrics=['acc',binary_PTA,Specificy,auc])# 调整像素值train_datagen = ImageDataGenerator(rescale=1./255)




history = model.fit_generator(
    train_generator,
    steps_per_epoch=150,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=100)

model.save('cats_and_dogs_small_1.h5')



# ACC
acc = history.history['acc']
val_acc = history.history['val_acc']

# BINARY_TPA  Sensitivity
sensitivity = history.history['binary_PTA']
val_sensitivity = history.history['val_binary_PTA']

#specificy
specificy = history.history['Specificy']
val_specificy = history.history['val_Specificy']

#auc
auc = history.history['auc']
val_auc = history.history['val_auc']

loss = history.history['loss']
val_loss = history.history['val_loss']




epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()


plt.plot(epochs, sensitivity, 'bo', label='Training sensitivity')
plt.plot(epochs, val_sensitivity, 'b', label='Validation sensitivity')
plt.title('Training and validation sensitivity')
plt.legend()
plt.figure()

plt.plot(epochs, specificy, 'bo', label='Training specificy')
plt.plot(epochs, val_specificy, 'b', label='Validation specificy')
plt.title('Training and validation specificy')
plt.legend()
plt.figure()


plt.plot(epochs, auc, 'bo', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.title('Training and validation auc')
plt.legend()
plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
