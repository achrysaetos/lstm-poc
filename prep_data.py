
# univariate data preparation
from numpy import array
from get_data import all_binance

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	inputs, outputs = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		inputs.append(seq_x)
		outputs.append(seq_y)
	return array(inputs), array(outputs)

# choose a window and a number of time steps
seq_size, n_steps = 100, 5
# define input sequence
raw_seq = all_binance[-seq_size:]
# split into samples
inputs, outputs = split_sequence(raw_seq, n_steps)