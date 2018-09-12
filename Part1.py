from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# Rescale images to fixed size
img_width, img_height = 150, 150

train_data_dir = 'train/train'
validation_data_dir = 'test/test1'

nb_train_samples = 1000
nb_validation_samples = 399
epochs = 10
batch_size = 32
                
    
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

# Define model

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

ADAM = Adam(lr=0.0001) #Optimiser

model.compile(loss='binary_crossentropy',
              optimizer=ADAM,
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose =1)



plt.rcParams['figure.figsize'] = (16,16) # Make the figures a bit bigger
base_path = 'test/test1/'

classes = {0:'Cat', 1:'Dog'} # Keras assigns integer ids (0,1,...) to class labels (cat,dog,...) alphabatecally
test_files = []

for x in os.listdir(base_path): 
    if (x.endswith('.jpg')):
        test_files.append(x)

for i,x in enumerate(test_files):
    img = image.load_img(base_path + x, target_size=(150, 150)) # Our trained network expects input images of 150x150 dimensions
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    output = model.predict(x)
    plt.subplot(6,4,i+1)
    plt.imshow(img)
    plt.title("Predicted: {}".format(classes[int(output[0][0])]))
    plt.axis('off')