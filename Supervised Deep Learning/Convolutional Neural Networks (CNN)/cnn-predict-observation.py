# Convolutional Neural Network

# =============================================================================
# Take a picture of your dog or your cat and use our CNN model to predict what it is.
# =============================================================================

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# =============================================================================
# # Step 1 - Convolution
# =============================================================================
# 3 dimentions image, make size 64 64. input_shape = (64, 64, 3)
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# =============================================================================
# # Step 2 - Pooling
# =============================================================================
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# =============================================================================
# # Step 3 - Flattening
# =============================================================================
classifier.add(Flatten())

# =============================================================================
# # Step 4 - Full connection
# =============================================================================
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# =============================================================================
# Apply Image Augmentation to enrich our dataset and prevent overfittiong. CNN needs to find variance in
# the pixels and this requires a lot of images.
# we can use Data augmentation instead  (will create many batches of our images,
# then each batch applies random transformations (rotating,shifting), allowing us to have many more training images)
# =============================================================================


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


#  Fitting the CNN to the images

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)










# =============================================================================
# Making new predictions
# =============================================================================

import numpy as np # to preprocess image that we will use
from keras.preprocessing import image

# must be same 64 64 size as we had in training set
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# our CNN input layer has 3 dimentions b/c colored image input_shape = (64, 64, 3)
# from 2d to 3d array (colored image rgb)
test_image = image.img_to_array(test_image)

# we will add add array of one input
# requires one more dimention tht coresponds to the batch (expects batch of inpputs)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

training_set.class_indices # mapping between cats and dogs and associated values

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
    
# =============================================================================
# Evaluation
# Evaluation was already made during the training with the validation set, 
# therefore k-Fold Cross Validation is not needed.
# =============================================================================


# =============================================================================
# Improving, Tuning
# The techniques to improve and tune a CNN model are the same as for ANNs.
# =============================================================================


