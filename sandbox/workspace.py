
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime
import csv
from numpy import array
from numpy import hstack

import sys
sys.path.insert(0, '..')
from config import binance_api_key, binance_api_secret
from data.get_data import get_binance_data

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'buy_next_period':False, 'sell_next_period':False}
states = {'vanilla':'hold', 'stacked':'hold', 'bidirectional':'hold'}
seq_size, n_steps = 180, 5
n_steps_in, n_steps_out = 5, 2
batch_size, num_epochs = 60, 200
open_seq = get_binance_data('BTCBUSD', '1m', download=True, col_name='open')
close_seq = get_binance_data('BTCBUSD', '1m', download=True, col_name='close')
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000, 'trades': 0}
w1 = {'cash': 1000000, 'coins': 0, 'value': 1000000, 'trades': 0}
w2 = {'cash': 1000000, 'coins': 0, 'value': 1000000, 'trades': 0}
w3 = {'cash': 1000000, 'coins': 0, 'value': 1000000, 'trades': 0}
wX = {'cash': 0, 'coins': 1000000/open_seq[0], 'value': 1000000, 'trades': 0}

def print_wallets(w1, w2, w3, wallet, wX, i):
    print(i, btc_price['close'], trade(w1))
    print(i, btc_price['close'], trade(w2))
    print(i, btc_price['close'], trade(w3))
    print(i, btc_price['close'], trade(wallet))
    print(i, btc_price['close'], trade(wX))
def print_states(x):
    if x%2 == 0: return 'sold'
    if x%2 == 1: return 'bought'

def trade(wallet, buy=False, sell=False):
    FEE = .001
    if sell:
        if wallet['coins'] > 0:
            wallet['cash'] = (1-FEE)*wallet['coins']*float(btc_price['close'])
            wallet['coins'] = 0
            wallet['trades'] += 1
    elif buy:
        if wallet['cash'] > 0:
            wallet['coins'] = ((1-FEE)*wallet['cash'])/float(btc_price['close'])
            wallet['cash'] = 0
            wallet['trades'] += 1
    wallet['value'] = wallet['coins']*float(btc_price['close'])+wallet['cash']
    return wallet

from data.prep_data import split_sequences_multivariate_multistep
from models.multivariate_multistep import vanilla, stacked, bidirectional
def simulate_multivariate_multistep(seq_size, n_steps_in, n_steps_out, batch_size, num_epochs, open_seq, close_seq, wallet, w1, w2, w3, wX):
    with open('4-18-2021.csv', mode='w') as multivariate_multistep:
        multivariate_multistep_writer = csv.writer(multivariate_multistep, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(len(open_seq)-seq_size):
            in_seq1 = array(close_seq[i:seq_size+i])
            in_seq2 = array(open_seq[i:seq_size+i])
            out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
            # convert to [rows, columns] structure
            in_seq1 = in_seq1.reshape((len(in_seq1), 1))
            in_seq2 = in_seq2.reshape((len(in_seq2), 1))
            out_seq = out_seq.reshape((len(out_seq), 1))
            # horizontally stack columns
            dataset = hstack((in_seq1, in_seq2, out_seq))
            # convert into input/output
            inputs, outputs = split_sequences_multivariate_multistep(dataset, n_steps_in, n_steps_out)
            btc_price["vanilla"] = vanilla(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2, batch_size, num_epochs)
            btc_price["stacked"] = stacked(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2, batch_size, num_epochs)
            btc_price["bidirectional"] = bidirectional(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2, batch_size, num_epochs)
            btc_price['open'], btc_price['close'] = open_seq[seq_size+i], close_seq[seq_size+i]
            if btc_price["vanilla"][0] < 0 and float(btc_price['close']) < float(btc_price['open']) and btc_price["vanilla"][1] < 0:
                trade(w1, sell=True)
                states['vanilla'] = 'sell'
            elif btc_price["vanilla"][0] > 0 and float(btc_price['close']) > float(btc_price['open']) and btc_price["vanilla"][1] > 0:
                trade(w1, buy=True)
                states['vanilla'] = 'buy'
            else: 
                states['vanilla'] = 'hold'
            if btc_price["stacked"][0] < 0 and float(btc_price['close']) < float(btc_price['open']) and btc_price["stacked"][1] < 0:
                trade(w2, sell=True)
                states['stacked'] = 'sell'
            elif btc_price["stacked"][0] > 0 and float(btc_price['close']) > float(btc_price['open']) and btc_price["stacked"][1] > 0:
                trade(w2, buy=True)
                states['stacked'] = 'buy'
            else: 
                states['stacked'] = 'hold'
            if btc_price["bidirectional"][0] < 0 and float(btc_price['close']) < float(btc_price['open']) and btc_price["bidirectional"][1] < 0:
                trade(w3, sell=True)
                states['bidirectional'] = 'sell'
            elif btc_price["bidirectional"][0] > 0 and float(btc_price['close']) > float(btc_price['open']) and btc_price["bidirectional"][1] > 0:
                trade(w3, buy=True)
                states['bidirectional'] = 'buy'
            else: 
                states['bidirectional'] = 'hold'
            if states['vanilla'] == states['stacked'] == states['bidirectional'] == 'sell':
                trade(wallet, sell=True)
            elif states['vanilla'] == states['stacked'] == states['bidirectional'] == 'buy':
                trade(wallet, buy=True)
            print(states)
            trade(wX)
            multivariate_multistep_writer.writerow([seq_size+i, w1['value'], w2['value'], w3['value'], wallet['value'], wX['value'], btc_price['open'], btc_price['close'], btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0], btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1], print_states(w1['trades']), print_states(w2['trades']), print_states(w3['trades']), print_states(wallet['trades'])])
            print_wallets(w1, w2, w3, wallet, wX, seq_size+i)

simulate_multivariate_multistep(seq_size, n_steps_in, n_steps_out, batch_size, num_epochs, open_seq, close_seq, wallet, w1, w2, w3, wX)
