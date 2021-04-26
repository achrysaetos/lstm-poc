
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime

import sys
sys.path.insert(0, '..')
from config import binance_api_key, binance_api_secret
from data.get_data import get_binance_data
from data.prep_data import split_sequence_univariate
from models.univariate import vanilla, stacked, bidirectional

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'change':False}
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}

def refresh():
    seq_size, n_steps = 360, 5
    batch_size, num_epochs = 60, 100
    raw_seq = get_binance_data('BTCBUSD', '1m')[-seq_size:]
    inputs, outputs = split_sequence_univariate(raw_seq, n_steps)
    btc_price["vanilla"] = vanilla(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs)
    btc_price["stacked"] = stacked(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs)
    btc_price["bidirectional"] = bidirectional(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs)
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

def simulate_univariate(msg):
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    seconds = now.strftime("%S")
    if int(seconds) % 60 == 1: 
        refresh()
        btc_price['change'] = True
    # define how to process incoming WebSocket messages
    if msg['e'] != 'error':
        btc_price['close'] = msg['c'] # current day close price
        if btc_price['change']:
            if float(btc_price['close']) > max([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                trade(wallet, sell=True)
            elif float(btc_price['close']) < min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                trade(wallet, buy=True)
            btc_price['change'] = not btc_price['change']
        print(timestamp, btc_price['close'], trade(wallet))
    else:
        btc_price['error'] = True

# init and start the WebSocket
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCBUSD', simulate_univariate)
bsm.start()

# stop websocket
# bsm.stop_socket(conn_key)
# properly terminate WebSocket
# reactor.stop()