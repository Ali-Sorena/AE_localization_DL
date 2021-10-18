import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from enum import auto
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# from tensorflow.keras import layers
# from tensorflow.python.ops.gen_array_ops import shape

# Locations:
fileloc = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/Data/Sensors CSV/"
madeUpDataSample = "MadeUpSamplesSet.csv"
masterData = "MasterData.csv"
targetfile = masterData

# Calling Database from CSV file
csv_file_path = os.path.join(fileloc, targetfile)
df = pd.read_csv(csv_file_path, delimiter=",")

values = df.values  # into numpy (taking indexes away)
values = values.astype('float32')  # Double check the type

scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(values)  # Normalizing(mn/var)

df_test_len = df.shape[0]
data_x, data_y = scaled_data[:, :-3], scaled_data[:, -2:]

separator = np.random.rand(len(scaled_data)) < 0.7  # choosing 80% of the dataset as training data
train = scaled_data[separator]
test = scaled_data[~separator]

print(train.shape)

train_x, train_y = train[:, :-3], train[:, -2:]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x, test_y = test[:, :-3], test[:, -2:]
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
# print("train")
# print(train_x.shape, type(train_x))
# print("Test")
# print(test_y.shape, type(test_y))

print(train_x[0].shape)
print(train_x[[0]].shape)

# magnitude of data
trx_shape_row = train_x.shape[0]
trx_shape_col = train_x.shape[1]

encoder_input = keras.Input(shape=(train_x.shape[1], train_x.shape[2]), name="tip")
# flatted = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(int(train_x.shape[0]/10), activation="relu")(encoder_input)  # Rectified linear unit

encoder = keras.Model(encoder_input, encoder_output, name="encoder")

decoder_input = keras.layers.Dense(trx_shape_col, activation="relu")(encoder_output)
decoder_output = keras.layers.Dense(trx_shape_col, activation="relu")(decoder_input)

opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)  # learning rate

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()

# print(trx_shape_row.shape)

# print(train_x.shape)
autoencoder.compile(opt, loss="mse")

history = autoencoder.fit(train_x, train_y, epochs=4, batch_size=1000, validation_split=0.1)

example = encoder.predict(test_x)

# example_output = autoencoder.predict(test_x[0])

# with tf.device(device):
# device = '/device:GPU:0'
