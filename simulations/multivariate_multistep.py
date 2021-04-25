
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
from data.prep_data import split_sequences_multivariate_multistep
from models.multivariate_multistep import vanilla, stacked, bidirectional

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'buy_next_period':False, 'sell_next_period':False, 'prev_period_price':0}
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}

def refresh():
    seq_size, n_steps_in, n_steps_out = 360, 5, 2
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
    inputs, outputs = split_sequences_multivariate_multistep(dataset, n_steps_in, n_steps_out)
    btc_price["vanilla"] = vanilla(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2)
    btc_price["stacked"] = stacked(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2)
    btc_price["bidirectional"] = bidirectional(inputs, outputs, n_steps_in, n_steps_out, in_seq1, in_seq2)
    print("Vanilla LSTM prediction:", btc_price["vanilla"])
    print("Stacked LSTM prediction:", btc_price["stacked"])
    print("Bidirectional prediction:", btc_price["bidirectional"])

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

def simulate_multivariate_multistep(msg):
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    seconds = now.strftime("%S")
    # define how to process incoming WebSocket messages
    if msg['e'] != 'error':
        btc_price['close'] = msg['c'] # current day close price
        print(timestamp, btc_price['close'], trade(wallet))
    else:
        btc_price['error'] = True
    if int(seconds) % 60 == 0:
        if btc_price['sell_next_period'] and float(btc_price['close']) < btc_price['prev_period_price'] and max([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]) < 0:
            print("Sell this period:", btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1], "< 0")
            trade(wallet, sell=True)
            btc_price['sell_next_period'] = False
        elif btc_price['buy_next_period'] and float(btc_price['close']) > btc_price['prev_period_price'] and min([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]) > 0:
            print("Buy this period:", btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1], "> 0")
            trade(wallet, buy=True)
            btc_price['buy_next_period'] = False
        else:
            print('No trade for this period -- insufficient confidence.')
    if int(seconds) % 60 == 1: 
        refresh()
        if max([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) < 0:
            btc_price['sell_next_period'] = True
            btc_price['prev_period_price'] = float(btc_price['close'])
            print("Sell next period:", btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0], "< 0")
        elif min([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) > 0:
            btc_price['buy_next_period'] = True
            btc_price['prev_period_price'] = float(btc_price['close'])
            print("Buy next period:", btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0], "> 0")
        else:
            print('No trade for next period -- unacceptable margin of error.')

# init and start the WebSocket
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCBUSD', simulate_multivariate_multistep)
bsm.start()

# stop websocket
# bsm.stop_socket(conn_key)
# properly terminate WebSocket
# reactor.stop()