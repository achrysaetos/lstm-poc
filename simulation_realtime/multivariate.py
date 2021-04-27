
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime
from numpy import array
from numpy import hstack

import sys
sys.path.insert(0, '..')
from config import binance_api_key, binance_api_secret
from data.get_data import get_binance_data
from data.prep_data import split_sequences_multivariate_multiinput, split_sequences_multivariate_multiparallel
from models.multivariate import multiinput_vanilla, multiinput_stacked, multiinput_bidirectional, multiparallel_vanilla, multiparallel_stacked, multiparallel_bidirectional

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'change':False}
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}

def refresh(version):
    seq_size, n_steps = 360, 5
    batch_size, num_epochs = 60, 100
    in_seq1 = array(get_binance_data('BTCBUSD', '1m', download=True, col_name='close')[-seq_size:])
    in_seq2 = array(get_binance_data('BTCBUSD', '1m', download=False, col_name='open')[-seq_size:])
    out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    dataset = hstack((in_seq1, in_seq2, out_seq))
    if version == "multiinput":
        inputs_multiinput, outputs_multiinput = split_sequences_multivariate_multiinput(dataset, n_steps)
        btc_price["vanilla"] = multiinput_vanilla(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
        btc_price["stacked"] = multiinput_stacked(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
        btc_price["bidirectional"] = multiinput_bidirectional(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
        print("Multi-input vanilla LSTM prediction:", btc_price["vanilla"])
        print("Multi-input stacked LSTM prediction:", btc_price["stacked"])
        print("Multi-input bidirectional prediction:", btc_price["bidirectional"])
    if version == "multiparallel":
        inputs_multiparallel, outputs_multiparallel = split_sequences_multivariate_multiparallel(dataset, n_steps)
        btc_price["vanilla"] = multiparallel_vanilla(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
        btc_price["stacked"] = multiparallel_stacked(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
        btc_price["bidirectional"] = multiparallel_bidirectional(inputs_multiparallel, outputs_multiparallel, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
        print("Multi-parallel vanilla LSTM prediction:", btc_price["vanilla"])
        print("Multi-parallel stacked LSTM prediction:", btc_price["stacked"])
        print("Multi-parallel bidirectional prediction:", btc_price["bidirectional"])

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

def simulate_multivariate_multiinput(msg):
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    seconds = now.strftime("%S")
    if msg['e'] != 'error':
        btc_price['close'] = msg['c'] # current day close price
        print(timestamp, btc_price['close'], trade(wallet))
    else:
        btc_price['error'] = True
    if int(seconds) % 60 == 1: 
        refresh("multiinput")
        if max([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]) < 0:
            trade(wallet, sell=True)
        elif min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]) > 0:
            trade(wallet, buy=True)

def simulate_multivariate_multiparallel(msg):
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    seconds = now.strftime("%S")
    if msg['e'] != 'error':
        btc_price['close'] = msg['c'] # current day close price
        print(timestamp, btc_price['close'], trade(wallet))
    else:
        btc_price['error'] = True
    if int(seconds) % 60 == 1: 
        refresh("multiparallel")
        if max([btc_price["vanilla"][2], btc_price["stacked"][2], btc_price["bidirectional"][2]]) < 0:
            trade(wallet, sell=True)
        elif min([btc_price["vanilla"][2], btc_price["stacked"][2], btc_price["bidirectional"][2]]) > 0:
            trade(wallet, buy=True)

# init and start the WebSocket
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCBUSD', simulate_multivariate_multiinput)
bsm.start()

# stop websocket
# bsm.stop_socket(conn_key)
# properly terminate WebSocket
# reactor.stop()