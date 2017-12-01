# Convolutional Neural Network

# =============================================================================
# Image recognition
# This is a Convolutional Neural Network that is able to detect various objects 
# in images. We implement this Deep Learning model to recognize a cat or a dog in a set of 
# pictures. However, this model can be reused to detect anything else - by simply changing 
# the pictures in the input folder. 
# 
# For example, you will be able to train the same model on a set of brain images, to detect if they 
# contain a tumor or not. But if you want to keep it fitted to cats and dogs, then you will literally 
# be able to a take a picture of your cat or your dog, and your model will predict which pet you have. 
# =============================================================================


# =============================================================================
# Data Preprocessing is Manual:
# Cannot exilictily add dependent var
# How to extract:
# Each image has name label, name label will be depended var vector 
# 
# Better soultuon:
# Prepare special structure, separate to test and train folder, AND separate folders with labels Cats, and Dogs which have images accordingly 
# 
# 10k images total, 8k train (4k dogs, 4k cats), 2k test
# 
# Hence data preprocessing is done manually
# 
# No categorical data, no need to encode, our data is already split.
# 
# Apply feature scaling before fitting
# =============================================================================


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # initialize CNNusing seq of layers 
from keras.layers import Conv2D # deal with images
from keras.layers import MaxPooling2D # pooling
from keras.layers import Flatten # convert pooled featuremaps into feature vetor
from keras.layers import Dense # add fully connected layers like in a classic NN

# Initialising the CNN
# CNN to classify images
classifier = Sequential() 

# =============================================================================
# convolution
# max pooling
# flattening
# full connection
# =============================================================================

# =============================================================================
# Step 1 - Convolution
# =============================================================================


# =============================================================================
# # input image 3d (rgb) array apply feature detectors(ex. 3x3 lernel) = feature map
# # applying feature detector: highest num in the feature map is where feature map 
# # was able to detect a specific feature in the input image

#(make image smaller, easier to processs). Lose some info, BUT we detect features that are integral 
#How well was the feature detected?  Feature map helps use preserve the main features (nose, eyes)
#Get ride of unnecessary info
#
#Multiple feature maps because we have many feature detectors (kernels)
#Network decides which what features to look for for certain categories. (keras initializes them randomly and then they are treated like weights which get updated during the backprop step.)


#ReLU rectifier activation function
#To increase non linearity, images are none linear, borders, colors,
#We risk in coming up with something leaner with CNNs so we apply rectifier function
#
#
#Ex:
#Instead of shadow going grey, darker, black, Linear progression from white to dark
#Make it same, not too accurate, approximate that area
# 
# =============================================================================
# num of feature detectors(filters), num rows of each (feature detector)filter, and # columns
# start with 32, 64, 128 feature detectors with 3x3 dimentions
# all of our images dont have the same size or format so convert them into same 
# format (so specify expected format before training), we are working with colored images = 3d array (rgb)
# select 3 colors but to make things faster use 64x64 format (otherwise choose 128 or 256 if on gpu)
# use rectifier activ func to not have negative values in feature maps (to have non-linearity)
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# =============================================================================
#  Step 2 - Pooling (Max Pooling) 
# =============================================================================
# =============================================================================
# apply max pooling on each of the feature maps to have spatial variance (similar to down sampling)
# ex:  Instead of shadow going grey, darker, black, Linear progression from white to dark
#      Make it same, not too accurate, approximate that area
# (Find max value in that kernel, record only that value)
# 
# ex. Since taking max, if eyes were rotated looking left instead of right, or tilted, it wont matter.
# And reducing size and parameters preventing overfitting 
# Getting ride of information of specifics of the feature, but retain most important part
# =============================================================================
# 2x2 kernel
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# =============================================================================
# # Adding a second convolutional layer and and Pooling (improve accuracy, make it deeper)
# =============================================================================
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# =============================================================================
# # Step 3 - Flattening
# =============================================================================
# =============================================================================
# # Each node will represent a specific feature of image, allowing us to eventually 
# # get spatial structure of input mage
# =============================================================================
# flatten all feature maps in pooling layer into vector
classifier.add(Flatten())

# =============================================================================
# # Step 4 - Full connection
# =============================================================================
# add layers
classifier.add(Dense(units = 128, activation = 'relu')) 
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# Compile the ANN (apply Stochastic gradient descent for back propagation)
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

# Fitting the CNN to the images

# apply the image augmentation, and resize images, batches of 32 images
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary') # cats and dogs, binary

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)