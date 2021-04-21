
# univariate multi-step lstm examples
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from get_data import get_binance_data # from get_data.py
from prep_data import split_sequence_multistep # from prep_data.py

# choose a window and a number of time steps
seq_size, n_steps_in, n_steps_out = 360, 5, 2
# define input sequence
raw_seq = get_binance_data('BTCBUSD', '1m')[-seq_size:]
# split into samples
inputs, outputs = split_sequence_multistep(raw_seq, n_steps_in, n_steps_out)

# univariate multi-step vector-output stacked lstm example
def vectoroutput_stacked(inputs, outputs, raw_seq, n_steps_in, n_steps_out):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
  model.add(LSTM(100, activation='relu'))
  model.add(Dense(n_steps_out))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=50, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps_in:])
  x_input = x_input.reshape((1, n_steps_in, n_features))
  yhat = model.predict(x_input, verbose=0)
  return yhat

from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# univariate multi-step encoder-decoder vanilla lstm example
def encoderdecoder_vanilla(inputs, outputs, raw_seq, n_steps_in, n_steps_out):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
  outputs = outputs.reshape((outputs.shape[0], outputs.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
  model.add(RepeatVector(n_steps_out))
  model.add(LSTM(100, activation='relu', return_sequences=True))
  model.add(TimeDistributed(Dense(1)))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=100, verbose=0)
  # demonstrate prediction
  x_input = array(raw_seq[-n_steps_in:])
  x_input = x_input.reshape((1, n_steps_in, n_features))
  yhat = model.predict(x_input, verbose=0)
  return yhat


# Univariate multi-step models for checkpoint gauging
print("Vector-output stacked LSTM prediction:", vectoroutput_stacked(inputs, outputs, raw_seq, n_steps_in, n_steps_out))

# Univariate multi-step models for sequence-to-sequence problems (ie. translation)
# print("Encoder-decoder vanilla LSTM prediction:", encoderdecoder_vanilla(inputs, outputs, raw_seq, n_steps_in, n_steps_out))