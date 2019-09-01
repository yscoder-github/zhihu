from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
from sklearn.datasets import load_iris
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

np.random.seed(0)
np.random.shuffle(iris_X)
np.random.seed(0)
np.random.shuffle(iris_y)

np.random.seed(7) # fix random seed for reproducibility
# create model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='softmax'))
# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(iris_X, iris_y, validation_split=0.20, epochs=80, batch_size=10)