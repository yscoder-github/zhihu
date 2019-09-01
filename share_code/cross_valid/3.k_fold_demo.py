from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def model1(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=200, batch_size=20, verbose=0)
    scores = model.evaluate(test_X, test_y, verbose=0)
    return scores 

def model2(train_X, train_y, test_X, test_y):
    model = Sequential()
    # model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, input_dim=4, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=200, batch_size=20, verbose=0)
    scores = model.evaluate(test_X, test_y, verbose=0)
    return scores

def model3(train_X, train_y, test_X, test_y):
    model = LinearRegression()
    model.fit(train_X , train_y)
    score = model.score(test_X , test_y)
    return score 

if __name__ == "__main__":
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    # fix random seed for reproducibility
    method1_acc = [] 
    method2_acc = [] 
    method3_acc = [] 
    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for train, test in kfold.split(iris_X, iris_y):
        # define 5-fold cross validation
        score1 = model1(iris_X[train], iris_y[train],iris_X[test], iris_y[test])
        score2 = model2(iris_X[train], iris_y[train],iris_X[test], iris_y[test]) 
        score3 = model3(iris_X[train], iris_y[train],iris_X[test], iris_y[test]) 
        method1_acc.append(score1[1])
        method2_acc.append(score2[1])
        method3_acc.append(score3)


    x_axix = [x for x in range(1, 6)] #开始画图
    plt.title('Result Analysis')
    plt.plot(x_axix, method1_acc, color='green', label='method1 accuracy')
    plt.plot(x_axix, method2_acc,  color='skyblue', label='method2 accuracy')
    plt.plot(x_axix, method3_acc, color='blue', label='method3 accuracy')
    plt.legend() # 显示图例
    
    plt.xlabel('iteration times')
    plt.ylabel('accuracy')
    plt.show()