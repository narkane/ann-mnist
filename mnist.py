from matplotlib import pyplot as plt
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np

np.random.seed(123)

# load preshuffled MNIST data into train and test setse
(xx_train, y_train), (x_test, y_test) = mnist.load_data()

# show first 3 numbers as images
for i in range(3):
    plt.imshow(xx_train[i])
    plt.show()

# recast dataset to have shape of (size, color_depth_channels, x_pixels, y_pixels)
# AKA added in a color_depth_channel value to shape
# ^^^^^CANCEL reshape(x_train.shape[0], 1, 28, 28) CANCEL^^^^^ #
# since Convolution2D was deprecated in tutorial form in keras2 the new model.add(dense())
# is working by multiplying 28 * 28 * 1 out to 784 nodes now by lowering from 3dims to 1dim and passing in that way
x_train = xx_train.reshape(xx_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
print(x_train.shape)
print(y_train.shape)

# convert dataset to float32 and normalize values from range [0,1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255    # normalizes
x_test /= 255

# y array is simply a 60000 array of decimal numbers
print(y_train[:10])
# [:1] = [5]

# convert test/train to 60000 BY 10 array where each entry has 10 slots and the appropriate value is binary high
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(y_train[:10])
# [:1] = [0 0 0 0 0 1 0 0 0 0]

# a model with a linear stack of layers
model = Sequential()
# 28x-pixels * 28y-pixels input (+1)
# multiplied by 512 = 401920 params
model.add(Dense(512, activation='relu', input_shape=(784,)))
# optomize out unneeded nodes
model.add(Dropout(0.25))
# last convolution of 512 nodes (+1)
# multiplied by new 512 = 262656 new params in this conv
model.add(Dense(512, activation='relu'))
# model.add(Flatten())
model.add(Dropout(0.5))
# previous 512 nodes (+1)
# multiplied by 10 output nodes = 5130 new params in final conv
model.add(Dense(10, activation='softmax'))
# total of 401920 + 262656 + 5130 = 669706
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=1,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

probability_model = keras.Sequential([model,
                                      keras.layers.Softmax()])

predictions = probability_model.predict(x_train)[:3]

# print('predictions shape: ', predictions.shape)
print('first 3 (5, 0, 4): ')
print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))
print(np.argmax(predictions[2]))

# model.predict(xx_train[2])
