
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime

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
def simulate_univariate(msg):
    seq_size, n_steps = 360, 5
    open_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='open')
    close_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='close')
    for i in range(len(open_seq)-seq_size):
        raw_seq = open_seq[i:seq_size+i]
        inputs, outputs = split_sequence_univariate(raw_seq, n_steps)
        btc_price["vanilla"] = vanilla(inputs, outputs, raw_seq, n_steps)
        btc_price["stacked"] = stacked(inputs, outputs, raw_seq, n_steps)
        btc_price["bidirectional"] = bidirectional(inputs, outputs, raw_seq, n_steps)
        btc_price['open'], btc_price['close'] = open_seq[i], close_seq[i]
        if float(btc_price['open']) > max([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
            trade(wallet, sell=True)
        elif float(btc_price['open']) < min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
            trade(wallet, buy=True)
        print(i, btc_price['open'], trade(wallet))

from data.prep_data import split_sequence_multistep
from models.univariate_multistep import vectoroutput_vanilla, vectoroutput_stacked, vectoroutput_bidirectional
def simulate_univariate_multistep():
    seq_size, n_steps_in, n_steps_out = 360, 5, 2
    open_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='open')
    close_seq = get_binance_data('BTCBUSD', '1m', download=False, col_name='close')
    for i in range(len(open_seq)-seq_size):
        raw_seq = open_seq[i:seq_size+i]
        inputs, outputs = split_sequence_multistep(raw_seq, n_steps_in, n_steps_out)
        btc_price["vanilla"] = vectoroutput_vanilla(inputs, outputs, raw_seq, n_steps_in, n_steps_out)
        btc_price["stacked"] = vectoroutput_stacked(inputs, outputs, raw_seq, n_steps_in, n_steps_out)
        btc_price["bidirectional"] = vectoroutput_bidirectional(inputs, outputs, raw_seq, n_steps_in, n_steps_out)
        btc_price['open'], btc_price['close'] = open_seq[i], close_seq[i]
        if float(btc_price['open']) > max([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) and float(btc_price['close']) > max([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
            trade(wallet, sell=True)
        elif float(btc_price['open']) < min([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) and float(btc_price['close']) < min([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
            trade(wallet, buy=True)
        print(i, btc_price['close'], trade(wallet))


"""
simulate_univariate()
simulate_univariate_multistep()
"""
