
import pandas as pd
import math
import os.path
import time
from numpy import array
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from config import binance_api_key, binance_api_secret # from config.py

binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
start_date = '19 Apr 2021'

def minutes_of_new_data(symbol, kline_size, data):
    if len(data) > 0: old = parser.parse(data["timestamp"].iloc[-1])
    else: old = datetime.strptime(start_date, '%d %b %Y')
    new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new

def get_binance_data(symbol, kline_size):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    if oldest_point == datetime.strptime(start_date, '%d %b %Y'): print('Downloading all available %s data for %s...' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s...' % (delta_min, symbol))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    data_df.to_csv(filename)
    print('Done!')
    return [float(x) for x in data_df['open'].tolist()]

# Univariate data preparation. Split into samples.
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
