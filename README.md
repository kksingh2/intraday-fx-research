# intraday fx research

backtesting a simple intraday strategy on eur/usd 5 minute data.

## the strategy

1. mark the high and low of the asian session (00:00 to 07:00 london)
2. during the london session, wait for price to spike through one of those levels (wick takes the level)
3. when a candle closes back inside the range, enter a trade in the other direction
4. stop a few pips beyond the wick, target the opposite side of the asian range

i tried a bunch of small variations (different sessions, different stop sizes, different filters) across the four research scripts.

## what i found

no edge once spread (0.6 pip) and slippage are included. the raw signal looked ok in sample but the in sample edge didnt hold out of sample. logged it as a thing i tried that didnt work.

## run

put a eur/usd 5 minute csv with columns `datetime, open, high, low, close` in the folder and run any of the python files.
