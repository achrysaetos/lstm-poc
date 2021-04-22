
from numpy import array

# split a univariate sequence into samples, for core univariate lstm models
def split_sequence_univariate(sequence, n_steps):
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

# split a univariate sequence into samples, for univariate multi-step lstm models
def split_sequence_multistep(sequence, n_steps_in, n_steps_out):
	inputs, outputs = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		inputs.append(seq_x)
		outputs.append(seq_y)
	return array(inputs), array(outputs)

# split a multivariate sequence into samples, for multivariate multi-input lstm models
def split_sequences_multivariate_multiinput(sequences, n_steps):
	inputs, outputs = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		inputs.append(seq_x)
		outputs.append(seq_y)
	return array(inputs), array(outputs)

# split a multivariate sequence into samples, for multivariate multi-parallel lstm models
def split_sequences_multivariate_multiparallel(sequences, n_steps):
	inputs, outputs = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		inputs.append(seq_x)
		outputs.append(seq_y)
	return array(inputs), array(outputs)

# split a multivariate sequence into samples, for multivariate multi-step lstm models
def split_sequences_multivariate_multistep(sequences, n_steps_in, n_steps_out):
	inputs, outputs = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		inputs.append(seq_x)
		outputs.append(seq_y)
	return array(inputs), array(outputs)
