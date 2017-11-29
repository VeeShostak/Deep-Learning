# Artificial Neural Network

# using k-fold-cross validation for evaluating
# setup k-cross-validation with hyperparameter tuning to tune
# if we detect overfitting apply Dropout Regularization to improve accuracy, reduce variance

# Evaluating, Improving, and Tuning our ANN 


# Notes:

# Dropout Regularization to reduce overfitting if needed
# =============================================================================
# Overfilltting is when your model was trained too much on the training set, that it becomes much less performant on the test set.
# We can observe this when we have a large difference in accuracies between training and test sets
# When overfitting happens you have a much higher accuracy on the training sets than the test sets.
# Another way to detect: when you observe a high variance when doing cross validation (correlations 
# that it learned too much do not apply to these test sets)
# =============================================================================

# =============================================================================
# Dropout Regularization to reduce overfitting if needed:
# At each iteration of the training, some neurons of the ANN are randomly disabled, to prevent 
# them from being too dependent on each other when they learn the correlations. Hence, 
# the ANN learns several independent correlations in the data, since each time there is 
# a diff configurations of neurons (neurons learn more independently and prevents them 
# from learning too much)
# =============================================================================


# hyperparameter tuning with Grid Search on our ANN: 
# =============================================================================
# Parameter tuning:
# We have parameters that are learned during training, (the weights)
# And we have parameters that stay fixed. (called hyper parameters (ex. Batch size, # epochs, #neurons in layers))
#                                                                   
# Parameter tuning consist of using the best values of these hyperparameters.
# We will use grid search, that will test several combinations of these values, and will eventually return the best selection. 
# 
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
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# ===
# START Encoding categorical data (converting into numbers)
# ===
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#  create dummyvars since our categorical variables are NOT comparable, cant be ranked
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# ===
# END Encoding categorical data (converting into numbers)
# ===

# Split the dataset into the Training set and Test set
# test size 0.2 to train ANN on 8k observations, and test performance on 2k observations
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling to standardize the range of independent variables or features of data (normalize)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ===========================
# Part 2 - Make the ANN
# ===========================

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # Dropout Regularization will randomly disable neurons in each layer

# Initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# classifier.add(Dropout(p = 0.1))

# Add the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1)) # fraction of neurons to drop at each iteration start with 10%

# Add the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# ===========================
# Part 3 - Making predictions and evaluating the model
# ===========================

# Predict the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)






# Predict a single new observation
# Predict if the customer with the following informations will leave the bank:

# 	•	Geography: France (France = o.0 = 0.0)
# 	•	Credit Score: 600
# 	•	Gender: Male      (Male = 1)
# 	•	Age: 40 years old
# 	•	Tenure: 3 years
# 	•	Balance: $60000
# 	•	Number of Products: 2
# 	•	Does this customer have a credit card ? Yes
# 	•	Is this customer an Active Member: Yes
# 	•	Estimated Salary: $50000

# our model was trained on X_train and it was scaled so use sc b/c it was scaled(normalized)
# new prediction must be made on the same scale
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#correct = cm[0][0] + cm[1][1]
#incorrect = cm[0][1] + cm[1][0]
#accuracy = (correct) / 2000 # tested on 2k











# ===========================
# Part 4 - Evaluating, Improving, and Tuning ANN (using k fold-cross validation)
# ===========================


# Evaluating the ANN
# NOTE: only preprocess data first
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# it will build (ANN classifier) exactly as we build it in part 2, but will be trained differently
# so copy the code, leaving out the fitting part
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# this classifier will not be trained on whole data set, but will be built on 
# k fold cross validation on 10 folds, each time measuring model performance on 1 test fold, (9 training)
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# The use of k fold cross validation is to return the relevant accuracy of the ANN.
#This will return the 10 accuracies that occur in k=10 cross validation
# @param n_jobs = number cpus to use (-1 means all cpus)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)


mean = accuracies.mean()
variance = accuracies.std()

# goal is to get low bias, low variance


# Improving the ANN
# if we detect overfitting apply Dropout Regularization
# Dropout Regularization to reduce overfitting if needed


# Tuning the ANN (k-cross-validation with hyperparameter tuning)
# NOTE: only preprocess data first
# Lets implement parameter tuning with Grid Search on our ANN: 
# =============================================================================
# Parameter tuning:
# We have parameters that are learned during training, (the weights)
# And we have parameters that stay fixed. (called hyper parameters (ex. Batch size, # epochs, #neurons in layers))
#                                                                   
# Parameter tuning consist of using the best values of these hyperparameters.
# We will use grid search, that will test several combinations of these values, and will eventually return the best selection. 
# 
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


# use optimizer param so we can tune it
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# hyperparameter tuning
parameters = {'batch_size': [25, 32], #powers of 2
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']} # both based on stochastic gradient descent (rmsprop for RNN)

# setup k-cross-validation with hyperparameter tuning
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# =============================================================================
# Results: 85% accuracy
# batch_size 25
# epochs 500
# optimizer: rmsprop
# =============================================================================
