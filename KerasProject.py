import numpy as np
import pandas as pd
from sklearn import neural_network, model_selection, metrics

# for deep learning modules
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import optimizers
from keras import metrics

"""STEP 2: load the data"""
"""STEP 3: shuffle the samples and split into train and test"""
# keras' built-in datasets already take care of train/test split
(train_in, train_out), (test_in, test_out) = mnist.load_data()

"""STEP 3.5: scale and reshape the data"""
# Data must be four-dimensional to can work with the Keras API
train_in = train_in.reshape(train_in.shape[0], train_in.shape[1], train_in.shape[2], 1)
test_in = test_in.reshape(test_in.shape[0], test_in.shape[1], test_in.shape[2], 1)
train_in = train_in.astype('float32')
test_in = test_in.astype('float32')

# Scaling
train_in /= 255
test_in /= 255

# using 10 here because that is the number of possible classifications (10 unique digits)
train_out = np_utils.to_categorical(train_out, 10)
test_out = np_utils.to_categorical(test_out, 10)

"""STEP 4: determine the CNN hyperparameters"""
# here, we must build each layer of the CNN

cnn = Sequential() # required for the layer-by-layer programming approach below

cnn.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(128, (1, 1), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1,1)))

cnn.add(Flatten()) # reduces 2d matrix to a column vector for hidden layer operations
cnn.add(Dense(128, activation='relu')) # hidden layer
# output layer, due to having 10 neurons (one for each possible digit classification) and softmax activation
cnn.add(Dense(10, activation='softmax')) 

method =  tf.keras.optimizers.Adam(learning_rate=0.001) # set training method and learning rate

"""STEP 5: train the ANN"""
# select type of loss (cross-entropy) and metric
# F1 score is not available but you can obtain precision and recall, then calculate F1 manually
cnn.compile(optimizer=method, loss='categorical_crossentropy', 
    metrics=['accuracy'])

cnn.fit(train_in, train_out, epochs=10, batch_size=128)

"""STEP 6: predict training outputs"""
pred_train_out = cnn.predict(train_in)

"""STEP 7: get the training score"""
train_score = cnn.evaluate(train_in, train_out, verbose=0)

"""STEP 8: predict training outputs"""
pred_test_out = cnn.predict(test_in)

"""STEP 9: get the training score"""
test_score = cnn.evaluate(test_in, test_out, verbose=0)

"""STEP 10: save etestuation results to a file"""
pd.DataFrame(np.array([train_score, test_score])).to_csv("score.csv", index = False, header = False)

"""STEP 11: display results to the console"""
print('Convolutional Neural Network (CNN) Implementation','\n-------------------------------------------')
print('Average training loss: ',"%.2f" %(100*train_score[0]),
    '%','\nAverage training accuracy: ',"%.2f" %(100 * train_score[1]),'%')
print('Average testing loss: ',"%.2f" %(100*test_score[0]),
    '%','\nAverage testing accuracy: ',"%.2f" %(100 * test_score[1]),'%')