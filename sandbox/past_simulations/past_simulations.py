
def simulate_univariate_multistep_realtime(msg):
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
        if btc_price['sell_next_period'] and float(btc_price['close']) > max([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
            trade(wallet, sell=True)
            print("Sell this period:", btc_price['close'], ">", btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1])
            btc_price['sell_next_period'] = False
        elif btc_price['buy_next_period'] and float(btc_price['close']) < min([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
            trade(wallet, buy=True)
            print("Buy this period:", btc_price['close'], "<", btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1])
            btc_price['buy_next_period'] = False
    if int(seconds) % 60 == 1: 
        refresh()
        if float(btc_price['close']) > max([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]):
            btc_price['sell_next_period'] = True
            print("Sell next period:", btc_price['close'], ">", btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0])
        elif float(btc_price['close']) < min([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]):
            btc_price['buy_next_period'] = True
            print("Buy next period:", btc_price['close'], "<", btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0])

def simulate_univariate_multistep_historical():
    seq_size, n_steps_in, n_steps_out = 360, 5, 2
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
            if float(btc_price['open']) > btc_price["vanilla"][0] and float(btc_price['close']) > btc_price["vanilla"][1]:
                trade(w1, sell=True)
            elif float(btc_price['open']) < btc_price["vanilla"][0] and float(btc_price['close']) < btc_price["vanilla"][1]:
                trade(w1, buy=True)
            if float(btc_price['open']) > btc_price["stacked"][0] and float(btc_price['close']) > btc_price["stacked"][1]:
                trade(w2, sell=True)
            elif float(btc_price['open']) < btc_price["stacked"][0] and float(btc_price['close']) < btc_price["stacked"][1]:
                trade(w2, buy=True)
            if float(btc_price['open']) > btc_price["bidirectional"][0] and float(btc_price['close']) > btc_price["bidirectional"][1]:
                trade(w3, sell=True)
            elif float(btc_price['open']) < btc_price["bidirectional"][0] and float(btc_price['close']) < btc_price["bidirectional"][1]:
                trade(w3, buy=True)
            if float(btc_price['open']) > max([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) and float(btc_price['close']) > max([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
                trade(wallet, sell=True)
            elif float(btc_price['open']) < min([btc_price["vanilla"][0], btc_price["stacked"][0], btc_price["bidirectional"][0]]) and float(btc_price['close']) < min([btc_price["vanilla"][1], btc_price["stacked"][1], btc_price["bidirectional"][1]]):
                trade(wallet, buy=True)
            trade(wX)
            univariate_multistep_writer.writerow([i, w1['value'], w2['value'], w3['value'], wallet['value'], wX['value']])
            print(i, btc_price['open'], trade(w1))
            print(i, btc_price['open'], trade(w2))
            print(i, btc_price['open'], trade(w3))
            print(i, btc_price['open'], trade(wallet))
            print(i, btc_price['open'], trade(wX))

def simulate_multivariate_multiinput_historical(seq_size, n_steps, batch_size, num_epochs, open_seq, close_seq, wallet, w1, w2, w3, wX):
    with open('multivariate_multiinput.csv', mode='w') as multivariate_multiinput:
        multivariate_multiinput_writer = csv.writer(multivariate_multiinput, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for i in range(len(open_seq)-seq_size):
            in_seq1 = array(close_seq[i:seq_size+i])
            in_seq2 = array(open_seq[i:seq_size+i])
            out_seq = array([in_seq1[i]-in_seq2[i] for i in range(len(in_seq1))])
            in_seq1 = in_seq1.reshape((len(in_seq1), 1))
            in_seq2 = in_seq2.reshape((len(in_seq2), 1))
            out_seq = out_seq.reshape((len(out_seq), 1))
            dataset = hstack((in_seq1, in_seq2, out_seq))
            inputs_multiinput, outputs_multiinput = split_sequences_multivariate_multiinput(dataset, n_steps)
            btc_price["vanilla"] = multiinput_vanilla(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
            btc_price["stacked"] = multiinput_stacked(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
            btc_price["bidirectional"] = multiinput_bidirectional(inputs_multiinput, outputs_multiinput, n_steps, in_seq1, in_seq2, batch_size, num_epochs)
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
            print_wallets(w1, w2, w3, wallet, wX, i)
