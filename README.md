# intraday fx research

backtesting one specific eur/usd trading idea on 5-minute price data.

## the idea

during the asian hours (overnight in london) the price usually stays in a small range. mark the high and low of that range.

then in the morning london session, watch for the price to spike just above that high or below that low. if it spikes through and then a candle closes back inside the range, take a trade in the opposite direction. the thinking is that the spike was "fake" - big players took out the obvious stops above the range, and the price is going back where it was.

- stop loss: a few pips beyond the wick of the spike
- take profit: the opposite side of the asian range

## what i tried

i ran about 150 small variations across the four research scripts (different session hours, different stop sizes, day-of-week filters, time-of-day filters, etc).

## what i found

in the raw price data it looked profitable. but once i added the spread the broker charges (0.6 pips, which is normal for eur/usd), the edge basically disappeared.

saving the code here so i remember next time not to chase the same idea without honest cost modelling.

## run

drop your own eur/usd 5 minute csv with columns `datetime, open, high, low, close` into the folder and run any of the python files.
