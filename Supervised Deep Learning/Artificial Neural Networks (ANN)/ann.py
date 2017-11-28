# Artificial Neural Network


# =============================================================================
#           Churn Modelling Problem
# You will be solving a data analytics challenge for a bank. 
# You will be given a dataset with a large sample of the bank's customers. 
# To make this dataset, the bank gathered information such as customer id, 
# credit score, gender, age, tenure(how long customer stayed with the bank), 
# balance, if the customer is active, has a credit card, etc. During a period of 
# 6 months, the bank observed if these customers left or stayed in the bank.
#
# Make an Artificial Neural Network that can predict, based on geo-demographical 
# and transactional information given above, if any individual customer will leave 
# the bank or stay (customer churn). Besides, you are asked to rank all the customers 
# of the bank, based on their probability of leaving. To do that, you will need to 
# use the right Deep Learning model, one that is based on a probabilistic approach 
# (sigmoid activation function for dependent variable). By applying your Deep Learning 
# model the bank may significantly reduce customer churn.
# =============================================================================

# ===========================
# Part 1 - Data Preprocessing
# ===========================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# create matrix x, of our features, independent variables to include
# exclude row# customerId, surename,
# ANN will give bigger weight to those independent vars that have most impact, so we will see
X = dataset.iloc[:, 3:13].values # matrix of features. indexes 3 - 12 (excludes 13). 
y = dataset.iloc[:, 13].values # dependent var (output to predict)


# ===
# START Encoding categorical data (converting into numbers)
# ===
# we have two categorical independent vars: Country(France Spain Germany), Gender (Male, Female)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# encode Country variable at index1 into 0,1,or 2
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 

# encode Gender variable at index2 into 0, or 1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# Dummy Variables: https://www.moresteam.com/WhitePapers/download/dummy-variables.pdf
# http://www.algosome.com/articles/dummy-variable-trap-regression.html
# Dummy variables assign the numbers ‘0’ and ‘1’ to indicate membership in any mutually exclusive and exhaustive category.

# Regression analysis treats all independent (X) variables in the analysis as 
# numerical. Numerical variables are interval or ratio scale variables 
# whose values are directly comparable
# our categorical variables are NOT comparable ((which is higher order? france, spain or grmany?) so we need dummy vars

# create dummy vars for this categorical variable (no need for gender)
onehotencoder = OneHotEncoder(categorical_features = [1]) # create dummy vars for index1 (country)
X = onehotencoder.fit_transform(X).toarray()
# now we have 12 indep vars, (3 new dummy vars: 3 new vars coresponding to the country)
# we need to remove one dummy var to avoid dummy variable trap
# Ex: with 3 columns worth of data, you're in effect encoding additional information that could be inferred with only two columns.  This additional information will cause problems with the regression analysis of the network
# hence: They're encoded values, in synthetic columns, used to represent the number of possible permutations of real categorical values
# (num of many dummy vars to represent single attribute variable is equal to the number of levels (categories) in that variable minus one.)

# don’t reate dummy seperate dummy var for gender to avoid dummy var trap: Including a dummy variable for each is redundant (of male is 0, female is 1, and vice-versa)
X = X[:, 1:] # take all features except first one
# ===
# END Encoding categorical data (converting into numbers)
# ===


# Split the dataset into the Training set and Test set
# test size 0.2 to train ANN on 8k observations, and test performance on 2k observations
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling to standardize the range of independent variables or features of data (normalize)
# If one of the features has a broad range of values, the distance will be governed by
# This particular feature. Therefore, the range of all features should be normalized so 
# that each feature contributes approximately proportionately to the final distance.

# due to intensive calculations and parallel execution 
# we need to ease calculations, dont want to have one independent var dominating a nother one 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# ===========================
# Part 2 - Make the ANN
# ===========================


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # to initialize ANN
from keras.layers import Dense # to create layers in ANN

# Initialising the ANN (we will define it as a sequence of layers or you can define a graph)
classifier = Sequential()

#1. Dense() function will randomly initialize the weights to small numbers close to 0 (but not 0)
#2. each feature in one input node: 11 inout vars => 11 input nodes
#3. farward propagation pass inpiuts from left to right where the neuron will be activated in way
#   where impact of each neurons activation is limited by the weights. (sum allweights, apply activation function)
#   using rectifier func for input layter and sigmoid for outer layer(to get probabilities of customer leaving or staying) 
#4. compare predicted result ot actual
#5. back propagation from right to left. update the weights according to how much they are responsible for he error
#   The learnign rate decides by how much we update the weights
#6. repeat steps 1-5 and update weights after each observation (row) (reinforcement learning)
#   Or: repeat steps 1-5 and updatee the weights only after a batch of observations (Batch learning)
#7. When the whole training (all rows of data) passed through the ANN that makes an epoch. Redo more epochs

# use stochastic gradient decscent


#Choose Number of nodes in hidden layer: as average number of nodes in hidden and output layer
#Or experiment with parameter tuning, ross validation techniques
# (11 + 1)/2 for hidden layer

# initialize weights randomly close to 0: kernel_initializer = 'uniform'

# using rectifier activation func for input layter and sigmoid for outer layer(to get probabilities of customer leaving or staying) 


# Add the input layer and the first hidden layer. (for initial layer specify the input nodes since we have no prev layer)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Add the second hidden layer (use prev layer as input nodes)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer (NOTEL: if 3 encoded categories for dependent variable need 3 nddes and softmax activator func)
# choose sigmoid just like in logistic regression
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compile the ANN (apply Stochastic gradient descent for back propagation)
# Stochastic gradient descent algorithm = adam. 
# loss function within adam alg, based on loss function that we need to optimize to find optimal weights 

# (for simple linear regression loss func is sum of squared errors. 
# in ML(perceptron NN and use sigmoid activation function you obtain a logistic regression model):
# looking into stochastic gradient descent loss func is NOT sum of squared errors but is a 
# logarithmic function called logarithmic loss )

# so use binary logarithmic loss func b/c (binary_entropy = dependent var has binary outcome, if i categories = categorical_crossentropy)
# criteria to evaluate our model metrics = ['accuracy'] (after weights updated, algo uses accuracy criterion to improve models performance) (when we fit accuracy will increase little by litle until rach top accuracy since we chose accuracy metric)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fit the ANN to the Training set (experimment with batch size and epochs)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)





# ===========================
# Part 3 - Making predictions and evaluating the model
# ===========================

# Predict the Test set results
y_pred = classifier.predict(X_test) # gives probability of leaving bank
y_pred = (y_pred > 0.5) # choose threshold to convert to true or false

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

correct = cm[0][0] + cm[1][1]
incorrect = cm[0][1] + cm[1][0]
accuracy = (correct) / 2000 # tested on 2k