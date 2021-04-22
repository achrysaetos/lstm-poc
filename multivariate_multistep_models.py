
# multivariate multi-step lstm examples
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from get_data import get_binance_data # from get_data.py
from prep_data import split_sequences_multivariate_multistep # from prep_data.py

# choose a window and a number of time steps
seq_size, n_steps_in, n_steps_out = 360, 5, 2
# define input sequence
in_seq1 = array(get_binance_data('BTCBUSD', '1m', download=True, col_name='high')[-seq_size:])
in_seq2 = array(get_binance_data('BTCBUSD', '1m', download=False, col_name='open')[-seq_size:])
out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# convert into input/output
inputs, outputs = split_sequences_multivariate_multistep(dataset, n_steps_in, n_steps_out)

# multivariate multi-step vanilla lstm example
def vanilla(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs.shape[2]
  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps_in:],in_seq2[-n_steps_in:])])
  x_input = x_input.reshape((1, n_steps_in, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0]), float(yhat[0][1])

# multivariate multi-step stacked lstm example
def stacked(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs.shape[2]
  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
  model.add(LSTM(100, activation='relu'))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps_in:],in_seq2[-n_steps_in:])])
  x_input = x_input.reshape((1, n_steps_in, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0]), float(yhat[0][1])

from keras.layers import Bidirectional
# multivariate multi-step bidirectional lstm example
def bidirectional(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs.shape[2]
  # define model
  model = Sequential()
  model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(n_steps_in, n_features)))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=200, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps_in:],in_seq2[-n_steps_in:])])
  x_input = x_input.reshape((1, n_steps_in, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0]), float(yhat[0][1])


# Multivariate multi-step models for 
print("Vanilla LSTM prediction:", vanilla(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2))
print("Stacked LSTM prediction:", stacked(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2))
print("Bidirectional prediction:", bidirectional(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2))