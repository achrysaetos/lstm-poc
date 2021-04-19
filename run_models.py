
# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from prep_data import inputs, outputs, raw_seq, n_steps

def vanilla(inputs, outputs, raw_seq, n_steps):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  print("Vanilla LSTM prediction:", yhat)

def stacked(inputs, outputs, raw_seq, n_steps):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
  model.add(LSTM(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  print("Stacked LSTM prediction:", yhat)

from keras.layers import Bidirectional
def bidirectional(inputs, outputs, raw_seq, n_steps):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  print("Bidirectional prediction:", yhat)

from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
def cnn(inputs, outputs, raw_seq, n_steps):
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
  model.add(LSTM(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=500, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_seq, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  print("Cnn LSTM prediction:", yhat)

from keras.layers import Flatten
from keras.layers import ConvLSTM2D
def conv(inputs, outputs, raw_seq, n_steps):
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
  model.fit(inputs, outputs, epochs=500, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  print("Conv LSTM prediction:", yhat)


# Basic LSTM models for sequential data
vanilla(inputs, outputs, raw_seq, n_steps)
stacked(inputs, outputs, raw_seq, n_steps)
bidirectional(inputs, outputs, raw_seq, n_steps)

# LSTM models generally used for 2D image/spatial data
# cnn(inputs, outputs, raw_seq, n_steps)
# conv(inputs, outputs, raw_seq, n_steps)