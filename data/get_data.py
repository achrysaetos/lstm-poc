
import pandas as pd
import math
import os.path
import time
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from config import binance_api_key, binance_api_secret # from config.py

binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
start_date = '22 Apr 2021'

def minutes_of_new_data(symbol, kline_size, data):
    if len(data) > 0: old = parser.parse(data["open_time"].iloc[-1])
    else: old = datetime.strptime(start_date, '%d %b %Y')
    new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new

def get_binance_data(symbol, kline_size, download = True, col_name = 'open'):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    if download:
        oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df)
        delta_min = (newest_point - oldest_point).total_seconds()/60
        if oldest_point == datetime.strptime(start_date, '%d %b %Y'): print('Downloading all available %s data for %s...' % (kline_size, symbol))
        else: print('Downloading %d minutes of new data available for %s...' % (delta_min, symbol))
        klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
        data = pd.DataFrame(klines, columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'can_be_ignored' ])
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
        data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
        else: data_df = data
        data_df.set_index('open_time', inplace=True)
        data_df.to_csv(filename)
        print('Done!')
    return [float(x) for x in data_df[col_name].tolist()]
