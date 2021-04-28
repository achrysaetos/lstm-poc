## LSTMs to predict crypto trends. POC.

*Warning: No one actually does LSTMs correctly when it comes to predicting prices. You have to build your own models from scratch if you want it to be accurate.*

*To start: pip3 install numpy, tensorflow, keras, pandas, python-binance, python-dateutil; set binance_api_key and binance_api_secret in config.py; initialize virtual env with `source env/bin/activate`; run `python3 run_models.py`.*

*Conclusion: Multivariate (multi-input) multi-step LSTMs seem to be the most accurate when it comes to predicting crypto trends. However, each of the core Vanilla/Stacked/Bidirectional LSTMs appear to have no general edge over the others, and therefore the best results are reached when we combine all three in one model. For future research, we might implement Multivariate (multi-parallel) multistep LSTMs, but multi-input LSTMs appear to be more accurate than multi-parallel LSTMs, anyway.*

**Why crypto?**
1) Volatility. Crypto prices change enough for strategies to actually make a difference.
2) No day trading cap. Individuals can programmatically buy and sell many times a day to optimize their strategies.

**Key Ideas:**\
We're tring to predict trends, *NOT* prices. In the end, we're actually looking for the probability a price will either go up/down by a large/small margin.

* What if we are wrong and the price drops? We lose money.
* What if we are wrong and the price rises? We neither gain/lose money.
* What if we are right and the price drops? We neither gain/lose money.
* What if we are right and the price rises? We make money.

Random walk tells us we can't predict prices in the short run. But if computers have a better than 50% chance of predicting trends correctly, trends can't be purely random.

* Can we predict the trend of many stocks in the long run? Yes, that's why portfolios exist.
* Can we predict the trend of one stock in the long run? No, too many factors might affect the stock at any point in time.
* Can we predict the trend of many stocks in the short run? Yes, if we're able to do so with one stock.
* Can we predict the trend of one stock in the short run? Yes, if efficient markets is even marginally true, it must be predictable.

*Eventually, no inputs will be arbitrary (window size, time steps, etc) -- they will all be determined in real time.*

**Notes:**\
*Alternatively, we might not even care about the trend of a price at a given moment in time if we know that it is inevitably going to fall or rise anyway. If we can predict the general trend of the price and the feasibility of how much it will fall or rise, we can buy or sell with confidence when the price reaches either of our limits.*

*All you really need to do is know when to sell -- just predict big drops, otherwise just hold by default.*