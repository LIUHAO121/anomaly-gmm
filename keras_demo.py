import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np

input_dim = 10
timestemps = 12

# Define Sequential model with 3 layers
model = keras.Sequential()
model.add(LSTM(100))
model.add(Dense(input_dim))

model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=[keras.metrics.MeanAbsoluteError()])

x_train = tf.random.normal([1000, timestemps, input_dim])
y_train = tf.random.normal([1000, input_dim])
model.fit(x_train,y_train,batch_size=62,epochs=3)
model.summary()
model.save('weights/temp')