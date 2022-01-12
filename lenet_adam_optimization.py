#lenet

import keras
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten 

from tensorflow.keras.optimizers import SGD, Adam

#load train and test dataset
def load_dataset():
  #load dataset
  (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

  #reshape dataset to have a single channel
  trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
  testX = testX.reshape(testX.shape[0], 28, 28, 1)

  #one hot encode target values
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)

  return trainX, trainY, testX, testY

#scale pixels
def prep_pixels(train, test):

  #convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')

  train_norm = train_norm/255.0
  test_norm = test_norm/255.0

  return train_norm, test_norm

#define cnn model

def define_model():
  model = Sequential()

  #first set of CONV - RELU - POOLING
  model.add(Conv2D(20, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  
  # , kernel_initializer='he_uniform'

  #second set of CONV - RELU - POOLING
  model.add(Conv2D(50, (5, 5), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dense(10, activation='softmax'))

  #compile model
  # opt = SGD(learning_rate=0.01, momentum=0.9)
  opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

#evaluate model using k-fold cross-validation

def evaluate_model(dataX, dataY, n_folds=5):
  scores, histories = list(), list()

  #prepare cross validatioin
  kfold = KFold(n_folds, shuffle=True, random_state=1)

  #enumerate splits
  for train_ix, test_ix in kfold.split(dataX):
    #define model
    model = define_model()

    #select rows for train and test
    trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

    #fit model
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=2)
    model.save('lenet.h5')
    #evaluate model
    _, acc = model.evaluate(testX, testY, verbose = 0)
    print('> %.3f' % (acc*100.0))

    #append scores
    scores.append(acc)
    histories.append(history)
  return scores, histories

#plot diagnostic learning curve

def summarize_diagnostics(histories):
  for i in range(len(histories)):
    #plot loss
    plt.subplot(211)
    plt.title('Cross entropy loss')
    plt.plot(histories[i].history['loss'], color='blue', label='train')
    plt.plot(histories[i].history['val_loss'], color='orange', label='test')

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification accuracy')
    plt.plot(histories[i].history['accuracy'], color='blue', label='train')
    plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
  plt.show()
  
  

#run the test harness for evaluating a model
def run_test_harness():
  #load dataset
  trainX, trainY, testX, testY = load_dataset()

  #prepare pixel data
  trainX, testX = prep_pixels(trainX, testX)

  #evaluate model
  scores, histories = evaluate_model(trainX, trainY)
  
  #learning curves
  summarize_diagnostics(histories)
  #summarize estimated performance
  print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
  

#entry point, run the test harness
run_test_harness()