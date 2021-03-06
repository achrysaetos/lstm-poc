
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import sys
sys.path.insert(0, '../data')
from get_data import get_binance_data
from prep_data import split_sequence_univariate

# choose a window and a number of time steps
seq_size, n_steps = 360, 5
# choose a batch size and a number of epochs
batch_size, num_epochs = 60, 100
# define input sequence
raw_seq = get_binance_data('BTCBUSD', '1m')[-seq_size:]
# split into samples
inputs, outputs = split_sequence_univariate(raw_seq, n_steps)

# Basic LSTM models for sequential data------------------------------------------------------------------------------------------

def vanilla(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(batch_size, activation='relu', input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=num_epochs, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

def stacked(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(batch_size, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
  model.add(LSTM(batch_size, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=num_epochs, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

from keras.layers import Bidirectional
def bidirectional(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(Bidirectional(LSTM(batch_size, activation='relu'), input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=num_epochs, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

# LSTM models generally used for 2D image/spatial data---------------------------------------------------------------------------

from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
def cnn(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs):
  # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
  n_features = 1
  n_seq = 2
  n_steps = 2
  inputs = inputs.reshape((inputs.shape[0], n_seq, n_steps, n_features))
  # define model
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
  model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(batch_size, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=num_epochs, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_seq, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

from keras.layers import Flatten
from keras.layers import ConvLSTM2D
def conv(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs):
  # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
  n_features = 1
  n_seq = 2
  n_steps = 2
  inputs = inputs.reshape((inputs.shape[0], n_seq, 1, n_steps, n_features))
  # define model
  model = Sequential()
  model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
  model.add(Flatten())
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=num_epochs, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])
