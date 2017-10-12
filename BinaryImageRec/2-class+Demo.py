
# coding: utf-8

# In[ ]:





# In[5]:


# Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3),activation = 'relu'))


# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolution layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a third convolution layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation= 'relu'))
classifier.add(Dense(output_dim = 1, activation= 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/home/demo/TrainingData',
        target_size=(64, 64),
        batch_size=10,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/demo/ValidationData',
        target_size=(64, 64),
        batch_size=5,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img( '/home/Prediction/325109432-4_3_Q25_00200559_000009.jpg',target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (result)
print (train_generator.class_indices)


# In[6]:


test_image = image.load_img( '/home/demo/Prediction/325047821-3_3_Q39_00204704_000015.jpg',target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (result)
print (train_generator.class_indices)


# In[7]:


test_image = image.load_img( '/home/demo/Prediction/325047821-3_3_Q39_00204704_000015.jpg',target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (result)
print (train_generator.class_indices)


# In[8]:


test_image = image.load_img( '/home/demo/Prediction/325109299-1_3_Q25_00205691_000031.jpg',target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (result)
print (train_generator.class_indices)

