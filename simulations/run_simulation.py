
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime

import sys
sys.path.insert(0, '..')
from config import binance_api_key, binance_api_secret
from data.get_data import get_binance_data
from data.prep_data import split_sequence
from data.univariate import vanilla, stacked, bidirectional

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'change':False}
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}

def btc_trade_history(msg):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_seconds = now.strftime("%S")
    if int(current_seconds) % 60 == 1: 
        seq_size, n_steps = 60, 3
        raw_seq = get_binance_data('BTCBUSD', '1m')[-seq_size:]
        inputs, outputs = split_sequence(raw_seq, n_steps)
        btc_price["vanilla"] = vanilla(inputs, outputs, raw_seq, n_steps)
        btc_price["stacked"] = stacked(inputs, outputs, raw_seq, n_steps)
        btc_price["bidirectional"] = bidirectional(inputs, outputs, raw_seq, n_steps)
        print("Vanilla LSTM prediction:", btc_price["vanilla"])
        print("Stacked LSTM prediction:", btc_price["stacked"])
        print("Bidirectional prediction:", btc_price["bidirectional"])
        btc_price['change'] = True
    # define how to process incoming WebSocket messages
    if msg['e'] != 'error':
        btc_price['close'] = msg['c'] # current day close price
        if btc_price['change']:
            if float(btc_price['close']) > max([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                if wallet['coins'] > 0:
                    wallet['cash'] = wallet['coins']*float(btc_price['close'])
                    wallet['coins'] = 0
                    print("Sell all at", btc_price['close'])
                else: print('Cannot sell.')
            elif float(btc_price['close']) < min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                if wallet['cash'] > 0:
                    wallet['coins'] = wallet['cash']/float(btc_price['close'])
                    wallet['cash'] = 0
                    print("Buy all at", btc_price['close'])
                else: print('Cannot buy.')
            else: print("Hold all at", btc_price['close'])
            btc_price['change'] = not btc_price['change']
        wallet['value'] = wallet['coins']*float(btc_price['close'])+wallet['cash']
        print(current_time, btc_price['close'], wallet)
    else:
        btc_price['error'] = True

# init and start the WebSocket
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCBUSD', btc_trade_history)
bsm.start()

# stop websocket
# bsm.stop_socket(conn_key)
# properly terminate WebSocket
# reactor.stop()