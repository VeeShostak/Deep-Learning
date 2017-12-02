# =============================================================================
# Recurrent Neural Network
# =============================================================================

# =============================================================================
# LSTM RNN (Long Short Term Memory Recurrent Neural Network). 
# (we use Long Short-Term Memory Networks (LSTMs) to solve problem of vanishing gradient)
# We will take the challenge to use it to predict the real Google stock price.
# =============================================================================

# =============================================================================
# # Part 1 - Data Preprocessing
# =============================================================================

# Importing the libraries
import numpy as np # for numpy arrays (only numpy arrays can be input in keras)
import matplotlib.pyplot as plt #visualize results at the end
import pandas as pd# import datasets

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # import data as a dataframe with numpy
# numpy array of one open price, iloc[:, 1:2] everything from left to right of column, index of colum we want (select open only)
training_set = dataset_train.iloc[:, 1:2].values 

# =============================================================================
# START Feature Scaling
# Feature scaling, standardisation vs normalisation
# When building RNN especially with sigmoid func in output layer Normalisation preferred
# =============================================================================
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
#get min and max price of the data to be able to apply normalisation formula, then for each price of trainning set compute formula(scale (normalise))
training_set_scaled = sc.fit_transform(training_set)
# =============================================================================
# END Feature Scaling
# =============================================================================

# =============================================================================
# Timesteps
# specify what RNN will need to remember to predict next price (wrong timestep can lead to overfitting)
# 60 time steps means that at each time t, the RNN will look at 60 timesteps(day) before time t,
# and based on the trends it is capturing during prev 60 time steps, it will try to predict the next output at time t+1
# 20 financial day sin a month, 60 = 3 months
# 
# 60 timesteps and one output at time t+1
# =============================================================================

# Creating a data structure with 60 timesteps and 1 output
X_train = [] # 60 prev days prices input
y_train = [] # next day price output
# last index of our observation 1257 (1258 excluded)
# (ex. first iteration X_train = 0-59, y_train=60) (what heppened prev 60 to predict t+1)
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) # append 60 prev prices
    y_train.append(training_set_scaled[i, 0]) # append t+1 price
# make them numpy arrays for RNN
X_train, y_train = np.array(X_train), np.array(y_train)

# =============================================================================
# reshaping the data, adding more dimentionality to our prev datastructure.
# we will add the unit: the number of predictors we can use to predict what we want.
# in this case these predictors are indicators. (our current indicator is open price)
# if you want to add a dimention in the numpy array use the reshaoe function
# =============================================================================
# Reshaping
# create new dimention.
# X_train.shape[0] number of observations (rows), X_train.shape[1] timesteps (columns), number of indicators = 1
# we are adding the open price as an indicator

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# =============================================================================
# # Part 2 - Building the RNN
# =============================================================================

# Importing the Keras libraries and packages
from keras.models import Sequential # NN object representing seq of layers
from keras.layers import Dense # to add output layer
from keras.layers import LSTM # add LSTM layers
from keras.layers import Dropout # add Dropout regularisation

# Initialising the RNN
# represents sequence of layers, prediting continuous value (price)
# (regression for predicting continuos value, classification for predicting a category)
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# paramse: neurons = 50, since we are adding another LSTM layer(stacked LSTM netwrok) True, input shape yimesteps and indicators
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # drop 20% (if 50 neurons, 10 ignored) during farward and back prop

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer, predicting value of price at time t+1
regressor.add(Dense(units = 1))

# Compiling the RNN (optimizer = RMSprop or adam). 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# batch size: batches of 32 stock prices
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)




# =============================================================================
# # Part 3 - Making the predictions and visualising the results
# =============================================================================

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# numpy array of one open price, iloc[:, 1:2] everything from left to right of column, index of colum we want (select open only)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# to predict price at t+1, we need 60 prev prices
# in order to get pst 60 prices we eed both training and test set

# dataset_total contains training and test sets. axis = 0 vertical concat
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# inputs start at: first financial day of jan 2017 - 60 financial days, ...
# (total - test = currentToPredictIndex. currentToPredictIndex-60 : up to last value)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# numpy dimention
inputs = inputs.reshape(-1,1)
# scale our inputs, with prev training our regressor was trained (keep test values as they are)
inputs = sc.transform(inputs)


# Creating a data structure with 60 timesteps and 1 output
X_test = [] # 60 prev days

# test set only contains data for first 20 financial days, 60+20= go up to 80 days
for i in range(60, 80):
    # for each observation get 60 prev days
    X_test.append(inputs[i-60:i, 0]) # append 60 prev prices

X_test = np.array(X_test)
# Reshaping
# create new dimention.
# X_train.shape[0] number of observations (rows), X_train.shape[1] timesteps (columns), number of indicators = 1
# we are adding the open price as an indicator
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_stock_price = regressor.predict(X_test)
# obtain actual values to disply
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()





# =============================================================================
# Evaluating
# =============================================================================

# =============================================================================
# For Regression, the way to evaluate the model performance is with a metric called 
# RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared 
# differences between the predictions and the real values.
# 
# 
# However for our specific Stock Price Prediction problem, evaluating the model with the 
# RMSE does not make much sense, since we are more interested in the directions taken by our 
# predictions, rather than the closeness of their values to the real stock price. We want to 
# check if our predictions follow the same directions as the real stock price and we don’t 
# really care whether our predictions are close to the real stock price. 
# =============================================================================

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# =============================================================================
# Then we consider dividing this RMSE by the range of the Google Stock Price values of January 
# 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is 
# more relevant since for example if you get an RMSE of 50, then this error would be very big 
# if the stock price values ranged around 100, but it would be very small if the stock price 
# values ranged around 10000.
# =============================================================================

rmse /= 800




# =============================================================================
# Improving
# =============================================================================


# =============================================================================
# Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it 
# would be even better to train it on the past 10 years.
# 
# Increasing the number of timesteps: the model remembered the stock prices from the 60 previous 
# financial days to predict the stock price of the next day. That’s because we chose a number of 
# 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 
# 120 timesteps (6 months).
# 
# Adding some other indicators: if you have the financial instinct that the stock price of some 
# other companies might be correlated to the one of Google, you could add this other stock price 
# as a new indicator in the training data.
# 
# Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
# Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of 
# neurones in the LSTM layers to respond better to the complexity of the problem and we chose to 
# include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more 
# neurones in each of the 4 (or more) LSTM layers.
# =============================================================================


# =============================================================================
# Tuning
# =============================================================================

# =============================================================================
# We can do some Parameter Tuning on the RNN model we implemented.
# 
# This time we are dealing with a Regression problem because we predict a continuous 
# outcome (the Google Stock Price).
# 
# 
# Parameter Tuning for Regression is the same as Parameter Tuning for Classification, 
# the only difference is that we have to replace:
# scoring = 'accuracy'  
# by:
# scoring = 'neg_mean_squared_error' 
# =============================================================================







