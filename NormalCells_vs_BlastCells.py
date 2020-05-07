# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 04:04:35 2020

@author: abhi0
"""

#CNN augmented with Image Data generator model

#Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.regularizers import l2
from keras import optimizers
from keras.layers import Dropout
from keras import initializers
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Defining the model -- CNN
classifier = Sequential()


#image_dims:
hor_dim=1024
Ver_dim=1024

#Convolution
classifier.add(Conv2D(128,(3, 3),strides=4, input_shape = (hor_dim,Ver_dim, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Dropout:
Dropout(rate=0.8)

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(bias_regularizer=l2(0.01),activity_regularizer=l2(0.1),units=128,activation = 'relu'))
initializers.he_normal(seed=42)
classifier.add(Dense(units = 1,activation ='sigmoid'))

#Compiling the CNN
#adam_opt=optimizers.Adam(learning_rate=0.01,beta_1=0.9, beta_2=0.99, amsgrad=True)
#optimizers.SGD(learning_rate=0.01, momentum=0.2, nesterov=False)
RMSprop_opt=optimizers.RMSprop(learning_rate=0.0001, rho=0.7)
classifier.compile(optimizer = RMSprop_opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

##Checkpoint:
#EarlyStopping(monitor='val_loss',verbose=1)
checkpoint=ModelCheckpoint(filepath='C:/Users/abhi0/OneDrive/Documents/Blood-plasma-identification/best_model.h5', monitor='val_loss', save_best_only=True)

train_datagen = ImageDataGenerator(rescale = 1./1024,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
                            'C:/Users/abhi0/OneDrive/Desktop/BlastCells_vs_NormalCells/TrainingSet',
                             target_size = (hor_dim, Ver_dim),
                             batch_size = 2,
                             class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
                            'C:/Users/abhi0/OneDrive/Desktop/BlastCells_vs_NormalCells/TestSet',
                             target_size = (hor_dim, Ver_dim),
                             batch_size = 2,
                             class_mode = 'binary')

history=classifier.fit_generator(training_set,
                         steps_per_epoch = 35,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 15,
                         callbacks=[checkpoint])


#Printing history:
print(history.history.keys())

#Accuracy history:
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

#loading the best model:
from keras.models import load_model

saved_model = load_model('best_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        "C:/Users/abhi0/OneDrive/Desktop/BlastCells_vs_NormalCells_Augmented/TestSet",
        target_size=(64,64),
        class_mode='binary',
        shuffle=False)

#Predicting probabilities from the model:
pred=saved_model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

from PIL import Image
image = Image.open('C:/Users/abhi0/OneDrive/Desktop/BlastCells_vs_NormalCells/TestSet/BlastCells/BPC (1).JPG')

from numpy import asarray

img = asarray(image)
img2=img[:64,:64]
test_image = img2.reshape((1,64,64,3))

img_class=saved_model.predict_classes(test_image)

print(img_class)
