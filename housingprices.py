from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import pandas as pd

#Step 1: Importing the dataset
#https://www.kaggle.com/shree1992/housedata
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:14].values
y = dataset.iloc[:, 1].values

#Step 2: Let's build our model!
model = Sequential()
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform'))

#Step 3: Compile, train, and save our trained model for later use.
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[metrics.mae])
model.fit(X, y, batch_size = 10, epochs = 200)
model.save("housing_price_model.h5")
