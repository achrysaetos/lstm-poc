
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from datetime import datetime

import sys
sys.path.insert(0, '..')
from config import binance_api_key, binance_api_secret
from data.get_data import get_binance_data
from data.prep_data import split_sequence_multistep
from models.univariate_multistep import vectoroutput_vanilla, vectoroutput_stacked, vectoroutput_bidirectional

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
btc_price = {'error':False, 'buy_next_period':False, 'sell_next_period':False, 'prev_period_price':0}
wallet = {'cash': 1000000, 'coins': 0, 'value': 1000000}

def refresh():
    seq_size, n_steps_in, n_steps_out = 360, 5, 2
    batch_size, num_epochs = 60, 100
    raw_seq = get_binance_data('BTCBUSD', '1m')[-seq_size:]
    inputs, outputs = split_sequence_multistep(raw_seq, n_steps_in, n_steps_out)
    btc_price["vanilla"] = vectoroutput_vanilla(inputs, outputs, raw_seq, n_steps_in, n_steps_out, batch_size, num_epochs)
    btc_price["stacked"] = vectoroutput_stacked(inputs, outputs, raw_seq, n_steps_in, n_steps_out, batch_size, num_epochs)
    btc_price["bidirectional"] = vectoroutput_bidirectional(inputs, outputs, raw_seq, n_steps_in, n_steps_out, batch_size, num_epochs)
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

def simulate_univariate_multistep(msg):
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
        if btc_price['sell_next_period'] and float(btc_price['close']) < btc_price['prev_period_price'] and float(btc_price['close']) > max([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
            print("Sell this period:", btc_price['close'], ">", btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1])
            trade(wallet, sell=True)
            btc_price['sell_next_period'] = False
        elif btc_price['buy_next_period'] and float(btc_price['close']) > btc_price['prev_period_price'] and float(btc_price['close']) < min([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
            print("Buy this period:", btc_price['close'], "<", btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1])
            trade(wallet, buy=True)
            btc_price['buy_next_period'] = False
        else:
            print('No trade for this period -- insufficient confidence.')
    if int(seconds) % 60 == 1: 
        refresh()
        if float(btc_price['close']) > max([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]):
            btc_price['sell_next_period'] = True
            btc_price['prev_period_price'] = float(btc_price['close'])
            print("Sell next period:", btc_price['close'], ">", btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0])
        elif float(btc_price['close']) < min([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]):
            btc_price['buy_next_period'] = True
            btc_price['prev_period_price'] = float(btc_price['close'])
            print("Buy next period:", btc_price['close'], "<", btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0])
        else:
            print('No trade for next period -- unacceptable margin of error.')

# init and start the WebSocket
bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCBUSD', simulate_univariate_multistep)
bsm.start()

# stop websocket
# bsm.stop_socket(conn_key)
# properly terminate WebSocket
# reactor.stop()