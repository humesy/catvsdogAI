from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16
import os

img_width, img_height = 150, 150

train_data_dir = 'train/train'
validation_data_dir = 'test/test1'

nb_train_samples = 1000
nb_validation_samples = 399
epochs = 5    
batch_size = 16         
outputs = 2
    
if K.image_data_format() == 'channels_first':
   input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = VGG16(weights='imagenet', include_top=False,input_shape = (img_width, img_height, 3)) 

for layer in model.layers[:11]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu", name = 'FC1')(x)
x = Dropout(0.25)(x)
x = Dense(512, activation="relu", name = 'FC2')(x)
predictions = Dense(outputs, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)

ADAM = Adam(lr=0.0001) #Optimiser

model_final.compile(loss='categorical_crossentropy',
              optimizer=ADAM,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') 

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') 

history = model_final.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose =1)


plt.rcParams['figure.figsize'] = (6,5)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title( "Accuracy ")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Error")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
plt.close()