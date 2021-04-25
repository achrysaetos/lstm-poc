
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
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}

def trade(wallet, buy=False, sell=False):
    if sell:
        if wallet['coins'] > 0:
            wallet['cash'] = wallet['coins']*float(btc_price['close'])
            wallet['coins'] = 0
            print("Sell all at", btc_price['close'])
        else: 
            print('Cannot sell.')
    elif buy:
        if wallet['cash'] > 0:
            wallet['coins'] = wallet['cash']/float(btc_price['close'])
            wallet['cash'] = 0
            print("Buy all at", btc_price['close'])
        else: 
            print('Cannot buy.')
    wallet['value'] = wallet['coins']*float(btc_price['close'])+wallet['cash']
    return wallet

from data.prep_data import split_sequence_univariate
from models.univariate import vanilla, stacked, bidirectional
def simulate_univariate():
    seq_size, n_steps = 360, 5
    open_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='open')
    close_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='close')
    wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w1 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w2 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w3 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    wX = {'cash': 0, 'coins': 1000000/open_seq[0], 'value': 1000000}
    with open('univariate.csv', mode='w') as univariate:
        univariate_writer = csv.writer(univariate, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(len(open_seq)-seq_size):
            raw_seq = open_seq[i:seq_size+i]
            inputs, outputs = split_sequence_univariate(raw_seq, n_steps)
            btc_price["vanilla"] = vanilla(inputs, outputs, raw_seq, n_steps)
            btc_price["stacked"] = stacked(inputs, outputs, raw_seq, n_steps)
            btc_price["bidirectional"] = bidirectional(inputs, outputs, raw_seq, n_steps)
            btc_price['open'], btc_price['close'] = open_seq[i], close_seq[i]
            if float(btc_price['open']) > btc_price["vanilla"]:
                trade(w1, sell=True)
            elif float(btc_price['open']) < btc_price["vanilla"]:
                trade(w1, buy=True)
            if float(btc_price['open']) > btc_price["stacked"]:
                trade(w2, sell=True)
            elif float(btc_price['open']) < btc_price["stacked"]:
                trade(w2, buy=True)
            if float(btc_price['open']) > btc_price["bidirectional"]:
                trade(w3, sell=True)
            elif float(btc_price['open']) < btc_price["bidirectional"]:
                trade(w3, buy=True)
            if float(btc_price['open']) > max([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                trade(wallet, sell=True)
            elif float(btc_price['open']) < min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                trade(wallet, buy=True)
            trade(wX)
            univariate_writer.writerow([i, w1['value'], w2['value'], w3['value'], wallet['value'], wX['value']])
            print(i, btc_price['open'], trade(w1))
            print(i, btc_price['open'], trade(w2))
            print(i, btc_price['open'], trade(w3))
            print(i, btc_price['open'], trade(wallet))
            print(i, btc_price['open'], trade(wX))

from data.prep_data import split_sequence_multistep
from models.univariate_multistep import vectoroutput_vanilla, vectoroutput_stacked, vectoroutput_bidirectional
def simulate_univariate_multistep():
    seq_size, n_steps_in, n_steps_out = 60, 3, 2
    open_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='open')
    close_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='close')
    wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w1 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w2 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w3 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    wX = {'cash': 0, 'coins': 1000000/open_seq[0], 'value': 1000000}
    with open('univariate_multistep.csv', mode='w') as univariate_multistep:
        univariate_multistep_writer = csv.writer(univariate_multistep, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(len(open_seq)-seq_size):
            raw_seq = open_seq[i:seq_size+i]
            inputs, outputs = split_sequence_multistep(raw_seq, n_steps_in, n_steps_out)
            btc_price["vanilla"] = vectoroutput_vanilla(inputs, outputs, raw_seq, n_steps_in, n_steps_out)
            btc_price["stacked"] = vectoroutput_stacked(inputs, outputs, raw_seq, n_steps_in, n_steps_out)
            btc_price["bidirectional"] = vectoroutput_bidirectional(inputs, outputs, raw_seq, n_steps_in, n_steps_out)
            btc_price['open'], btc_price['close'] = open_seq[i], close_seq[i]
            if float(btc_price['open']) > btc_price["vanilla"][0] and btc_price['close'] < btc_price['open'] and float(btc_price['close']) > btc_price["vanilla"][1]:
                trade(w1, sell=True)
            elif float(btc_price['open']) < btc_price["vanilla"][0] and btc_price['close'] > btc_price['open'] and float(btc_price['close']) < btc_price["vanilla"][1]:
                trade(w1, buy=True)
            if float(btc_price['open']) > btc_price["stacked"][0] and btc_price['close'] < btc_price['open'] and float(btc_price['close']) > btc_price["stacked"][1]:
                trade(w2, sell=True)
            elif float(btc_price['open']) < btc_price["stacked"][0] and btc_price['close'] > btc_price['open'] and float(btc_price['close']) < btc_price["stacked"][1]:
                trade(w2, buy=True)
            if float(btc_price['open']) > btc_price["bidirectional"][0] and btc_price['close'] < btc_price['open'] and float(btc_price['close']) > btc_price["bidirectional"][1]:
                trade(w3, sell=True)
            elif float(btc_price['open']) < btc_price["bidirectional"][0] and btc_price['close'] > btc_price['open'] and float(btc_price['close']) < btc_price["bidirectional"][1]:
                trade(w3, buy=True)
            if float(btc_price['open']) > max([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) and btc_price['close'] < btc_price['open'] and float(btc_price['close']) > max([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
                trade(wallet, sell=True)
            elif float(btc_price['open']) < min([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) and btc_price['close'] > btc_price['open'] and float(btc_price['close']) < min([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
                trade(wallet, buy=True)
            trade(wX)
            univariate_multistep_writer.writerow([i, w1['value'], w2['value'], w3['value'], wallet['value'], wX['value']])
            print(i, btc_price['open'], trade(w1))
            print(i, btc_price['open'], trade(w2))
            print(i, btc_price['open'], trade(w3))
            print(i, btc_price['open'], trade(wallet))
            print(i, btc_price['open'], trade(wX))

from data.prep_data import split_sequences_multivariate_multiinput
from models.multivariate import multiinput_vanilla, multiinput_stacked, multiinput_bidirectional
def simulate_multivariate_multiinput():
    seq_size, n_steps = 60, 3
    close_seq = array(get_binance_data('BTCBUSD', '1m', download=True, col_name='close'))
    open_seq = array(get_binance_data('BTCBUSD', '1m', download=False, col_name='open'))
    wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w1 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w2 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w3 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    wX = {'cash': 0, 'coins': 1000000/open_seq[0], 'value': 1000000}
    with open('multivariate_multiinput.csv', mode='w') as multivariate_multiinput:
        multivariate_multiinput_writer = csv.writer(multivariate_multiinput, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(len(open_seq)-seq_size):
            in_seq1 = close_seq[i:seq_size+i]
            in_seq2 = open_seq[i:seq_size+i]
            out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
            in_seq1 = in_seq1.reshape((len(in_seq1), 1))
            in_seq2 = in_seq2.reshape((len(in_seq2), 1))
            out_seq = out_seq.reshape((len(out_seq), 1))
            dataset = hstack((in_seq1, in_seq2, out_seq))
            inputs_multiinput, outputs_multiinput = split_sequences_multivariate_multiinput(dataset, n_steps)
            btc_price["vanilla"] = multiinput_vanilla(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2)
            btc_price["stacked"] = multiinput_stacked(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2)
            btc_price["bidirectional"] = multiinput_bidirectional(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2)
            btc_price['open'], btc_price['close'] = open_seq[i], close_seq[i]
            if btc_price["vanilla"] < 0:
                trade(w1, sell=True)
            elif btc_price["vanilla"] > 0:
                trade(w1, buy=True)
            if btc_price["stacked"] < 0:
                trade(w2, sell=True)
            elif btc_price["stacked"] > 0:
                trade(w2, buy=True)
            if btc_price["bidirectional"] < 0:
                trade(w3, sell=True)
            elif btc_price["bidirectional"] > 0:
                trade(w3, buy=True)
            if max([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]) < 0:
                trade(wallet, sell=True)
            elif min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]) > 0:
                trade(wallet, buy=True)
            trade(wX)
            multivariate_multiinput_writer.writerow([i, w1['value'], w2['value'], w3['value'], wallet['value'], wX['value']])
            print(i, btc_price['open'], trade(w1))
            print(i, btc_price['open'], trade(w2))
            print(i, btc_price['open'], trade(w3))
            print(i, btc_price['open'], trade(wallet))
            print(i, btc_price['open'], trade(wX))

from data.prep_data import split_sequences_multivariate_multiparallel
from models.multivariate import multiparallel_vanilla, multiparallel_stacked, multiparallel_bidirectional
def simulate_multivariate_multiparallel():
    seq_size, n_steps = 60, 3
    close_seq = array(get_binance_data('BTCBUSD', '1m', download=True, col_name='close'))
    open_seq = array(get_binance_data('BTCBUSD', '1m', download=False, col_name='open'))
    wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w1 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w2 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    w3 = {'cash': 1000000, 'coins': 0, 'value': 1000000}
    wX = {'cash': 0, 'coins': 1000000/open_seq[0], 'value': 1000000}
    with open('multivariate_multiparallel.csv', mode='w') as multivariate_multiparallel:
        multivariate_multiparallel_writer = csv.writer(multivariate_multiparallel, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(len(open_seq)-seq_size):
            in_seq1 = close_seq[i:seq_size+i]
            in_seq2 = open_seq[i:seq_size+i]
            out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
            in_seq1 = in_seq1.reshape((len(in_seq1), 1))
            in_seq2 = in_seq2.reshape((len(in_seq2), 1))
            out_seq = out_seq.reshape((len(out_seq), 1))
            dataset = hstack((in_seq1, in_seq2, out_seq))
            inputs_multiparallel, outputs_multiparallel = split_sequences_multivariate_multiparallel(dataset, n_steps)
            btc_price["vanilla"] = multiparallel_vanilla(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2)
            btc_price["stacked"] = multiparallel_stacked(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2)
            btc_price["bidirectional"] = multiparallel_bidirectional(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2)
            btc_price['open'], btc_price['close'] = open_seq[i], close_seq[i]
            if btc_price["vanilla"][2] < 0:
                trade(w1, sell=True)
            elif btc_price["vanilla"][2] > 0:
                trade(w1, buy=True)
            if btc_price["stacked"][2] < 0:
                trade(w2, sell=True)
            elif btc_price["stacked"][2] > 0:
                trade(w2, buy=True)
            if btc_price["bidirectional"][2] < 0:
                trade(w3, sell=True)
            elif btc_price["bidirectional"][2] > 0:
                trade(w3, buy=True)
            if max([btc_price["vanilla"][2], btc_price["stacked"][2], btc_price["bidirectional"][2]]) < 0:
                trade(wallet, sell=True)
            elif max([btc_price["vanilla"][2], btc_price["stacked"][2], btc_price["bidirectional"][2]]) > 0:
                trade(wallet, buy=True)
            trade(wX)
            multivariate_multiparallel_writer.writerow([i, w1['value'], w2['value'], w3['value'], wallet['value'], wX['value']])
            print(i, btc_price['open'], trade(w1))
            print(i, btc_price['open'], trade(w2))
            print(i, btc_price['open'], trade(w3))
            print(i, btc_price['open'], trade(wallet))
            print(i, btc_price['open'], trade(wX))


"""
simulate_univariate()
simulate_univariate_multistep()
simulate_multivariate_multiinput()
simulate_multivariate_multiparallel()
"""
