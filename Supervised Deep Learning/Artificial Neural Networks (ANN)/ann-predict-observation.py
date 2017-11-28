# Artificial Neural Network

# =============================================================================
# Use our ANN model to predict if the customer with the following informations will leave the bank: 
# 	•	Geography: France
# 	•	Credit Score: 600
# 	•	Gender: Male
# 	•	Age: 40 years old
# 	•	Tenure: 3 years
# 	•	Balance: $60000
# 	•	Number of Products: 2
# 	•	Does this customer have a credit card ? Yes
# 	•	Is this customer an Active Member: Yes
# 	•	Estimated Salary: $50000
# So should we say goodbye to that customer ?
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

# Initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Add the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

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