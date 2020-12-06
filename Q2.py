import numpy as np
from tensorflow import keras
import Q1
# from tensorflow.keras import layers

def extract_data() :
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = x_train.reshape(x_train.shape[0] ,x_train.shape[1]*x_train.shape[2] )
    x_test = x_test.reshape(x_test.shape[0] ,x_test.shape[1]*x_test.shape[2] )

    end = int(0.3 * x_train.shape[0])
    return x_train[:end] , y_train[:end] , x_test , y_test

X_train , y_train , X_test , y_test = extract_data()
print(X_train.shape)

model = Q1.MyNeuralNetwork(5 , [784,256,128,64,10] , 'sigmoid' , 0.1 , 'zero' , 10000 ,  10)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)

print(acc)