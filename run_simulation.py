
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime

from config import binance_api_key, binance_api_secret # from config.py
from get_data import get_binance_data
from prep_data import split_sequence
from run_models import vanilla, stacked, bidirectional

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'change':False}

def btc_trade_history(msg):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_seconds = now.strftime("%S")
    if int(current_seconds) % 60 == 1: 
        seq_size, n_steps = 360, 5
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
                print("Sell all at", btc_price['close'])
            elif float(btc_price['close']) < min([btc_price["vanilla"], btc_price["stacked"], btc_price["bidirectional"]]):
                print("Buy all at", btc_price['close'])
            else:
                print("Hold all at", btc_price['close'])
            btc_price['change'] = not btc_price['change']
        print(current_time, btc_price['close'])
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