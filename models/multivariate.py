
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import sys
sys.path.insert(0, '../data')
from get_data import get_binance_data
from prep_data import split_sequences_multivariate_multiinput, split_sequences_multivariate_multiparallel

# choose a window and a number of time steps
seq_size, n_steps = 360, 5
# define input sequence
in_seq1 = array(get_binance_data('BTCBUSD', '1m', download=True, col_name='close')[-seq_size:])
in_seq2 = array(get_binance_data('BTCBUSD', '1m', download=False, col_name='open')[-seq_size:])
out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# convert into input/output
inputs_multiinput, outputs_multiinput = split_sequences_multivariate_multiinput(dataset, n_steps)
inputs_multiparallel, outputs_multiparallel = split_sequences_multivariate_multiparallel(dataset, n_steps)

# Multivariate multi-input models for triangulating predictions------------------------------------------------------------------

def multiinput_vanilla(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs_multiinput.shape[2]
  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs_multiinput, outputs_multiinput, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

def multiinput_stacked(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs_multiinput.shape[2]
  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
  model.add(LSTM(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs_multiinput, outputs_multiinput, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

from keras.layers import Bidirectional
def multiinput_bidirectional(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs_multiinput.shape[2]
  # define model
  model = Sequential()
  model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs_multiinput, outputs_multiinput, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])

# Multivariate multi-parallel models for predicting multiple factors-------------------------------------------------------------

def multiparallel_vanilla(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs_multiparallel.shape[2]
  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
  model.add(Dense(n_features))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs_multiparallel, outputs_multiparallel, epochs=400, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b,a-b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0]), float(yhat[0][1]), float(yhat[0][2])

def multiparallel_stacked(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs_multiparallel.shape[2]
  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
  model.add(LSTM(100, activation='relu'))
  model.add(Dense(n_features))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs_multiparallel, outputs_multiparallel, epochs=400, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b,a-b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0]), float(yhat[0][1]), float(yhat[0][2])

from keras.layers import Bidirectional
def multiparallel_bidirectional(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs_multiparallel.shape[2]
  # define model
  model = Sequential()
  model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(n_steps, n_features)))
  model.add(Dense(n_features))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs_multiparallel, outputs_multiparallel, epochs=400, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b,a-b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0]), float(yhat[0][1]), float(yhat[0][2])
